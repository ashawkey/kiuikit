import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

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

class HashGrid(nn.Module):
    def __init__(self, 
                 input_dim=3,
                 num_levels=16, 
                 level_dim=2, 
                 log2_hashmap_size=19, 
                 base_resolution=16, 
                 desired_resolution=2048, 
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