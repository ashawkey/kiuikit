import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

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

class _TruncExp(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply

# cosine lr scheduler with warm up
# ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/optimization.py#L154C1-L185C54
def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)