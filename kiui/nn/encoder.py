import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from kiui.nn import MLP

class HashGridEncoder(nn.Module):
    def __init__(self, 
                 input_dim=3,
                 num_levels=16,
                 level_dim=2,
                 log2_hashmap_size=18, 
                 base_resolution=16, 
                 desired_resolution=1024, 
                 interpolation='linear'
                 ):
        super().__init__()
        import tinycudann as tcnn
        self.encoder = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": np.exp2(np.log2(desired_resolution / num_levels) / (num_levels - 1)),
                "interpolation": "Smoothstep" if interpolation == 'smoothstep' else "Linear",
            },
            dtype=torch.float32,
        )
        self.input_dim = input_dim
        self.output_dim = self.encoder.n_output_dims # patch
    
    def forward(self, x, bound=1):
        return self.encoder((x + bound) / (2 * bound))


class FrequencyEncoder(nn.Module):
    def __init__(self, 
                 input_dim=3,
                 output_dim=32,
                 n_frequencies=12,
                 ):
        super().__init__()
        import tinycudann as tcnn
        self.encoder = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": n_frequencies,
            },
            dtype=torch.float32,
        )
        self.implicit_mlp = MLP(self.encoder.n_output_dims, output_dim, 128, 5, bias=True)
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x, **kwargs):
        return self.implicit_mlp(self.encoder(x))
    

class VMEncoder(nn.Module):
    def __init__(self, 
                 input_dim=3,
                 output_dim=32,
                 resolution=256,
                 mode='cat', # cat or sum
                 ):
        super().__init__()

        self.C_mat = nn.Parameter(torch.randn(3, output_dim, resolution, resolution))
        self.C_vec = nn.Parameter(torch.randn(3, output_dim, resolution, 1))
        torch.nn.init.kaiming_normal_(self.C_mat)
        torch.nn.init.kaiming_normal_(self.C_vec)
        
        self.mat_ids = [[0, 1], [0, 2], [1, 2]]
        self.vec_ids = [2, 1, 0]

        self.mode = mode
        self.input_dim = input_dim
        self.output_dim = output_dim * 3 if mode == 'cat' else output_dim
    
    def forward(self, x, bound=1):

        N = x.shape[0]
        x = x / bound # to [-1, 1]

        mat_coord = torch.stack((x[..., self.mat_ids[0]], x[..., self.mat_ids[1]], x[..., self.mat_ids[2]])).view(3, -1, 1, 2) # [3, N, 1, 2]
        vec_coord = torch.stack((x[..., self.vec_ids[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(3, -1, 1, 2) # [3, N, 1, 2], fake 2d coord
        
        feats = [
            F.grid_sample(self.C_mat[[0]], mat_coord[[0]], align_corners=False).view(-1, N) * F.grid_sample(self.C_vec[[0]], vec_coord[[0]], align_corners=False).view(-1, N),
            F.grid_sample(self.C_mat[[1]], mat_coord[[1]], align_corners=False).view(-1, N) * F.grid_sample(self.C_vec[[1]], vec_coord[[1]], align_corners=False).view(-1, N),
            F.grid_sample(self.C_mat[[2]], mat_coord[[2]], align_corners=False).view(-1, N) * F.grid_sample(self.C_vec[[2]], vec_coord[[2]], align_corners=False).view(-1, N),
        ] # 3 x [R, N]

        if self.mode == 'cat':
            feats = torch.cat(feats, dim=0) # [3R, N]
        else: # sum
            feats = torch.stack(feats, dim=0).sum(dim=0) # [R, N]
    
        feats = feats.transpose(0, 1).contiguous() # [N, R]
        return feats

    def tv_loss(self):
        loss = ((self.C_mat[:, :, 1:, :] - self.C_mat[:, :, :-1, :]) ** 2).mean() + \
               ((self.C_mat[:, :, :, 1:] - self.C_mat[:, :, :, :-1]) ** 2).mean() + \
               ((self.C_vec[:, :, 1:, :] - self.C_vec[:, :, :-1, :]) ** 2).mean()
        return loss

class TriplaneEncoder(nn.Module):
    def __init__(self, 
                 input_dim=3,
                 output_dim=32,
                 resolution=256,
                 ):
        super().__init__()

        self.C_mat = nn.Parameter(torch.randn(3, output_dim, resolution, resolution))
        torch.nn.init.kaiming_normal_(self.C_mat)
        
        self.mat_ids = [[0, 1], [0, 2], [1, 2]]

        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x, bound=1):

        N = x.shape[0]
        x = x / bound # to [-1, 1]

        mat_coord = torch.stack((x[..., self.mat_ids[0]], x[..., self.mat_ids[1]], x[..., self.mat_ids[2]])).view(3, -1, 1, 2) # [3, N, 1, 2]

        feat = F.grid_sample(self.C_mat[[0]], mat_coord[[0]], align_corners=False).view(-1, N) + \
               F.grid_sample(self.C_mat[[1]], mat_coord[[1]], align_corners=False).view(-1, N) + \
               F.grid_sample(self.C_mat[[2]], mat_coord[[2]], align_corners=False).view(-1, N) # [r, N]

        # density
        feat = feat.transpose(0, 1).contiguous() # [N, C]
        return feat

    def tv_loss(self):
        loss = ((self.C_mat[:, :, 1:, :] - self.C_mat[:, :, :-1, :]) ** 2).mean() + \
               ((self.C_mat[:, :, :, 1:] - self.C_mat[:, :, :, :-1]) ** 2).mean()
        return loss