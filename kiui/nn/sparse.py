from typing import List, Tuple, Union
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import torchsparse
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.utils.collate import sparse_collate, sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

class SparseConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: int = 1,
    ):
        super().__init__(
            spnn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
        )

class SparseResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size, dilation=dilation, stride=stride),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
            spnn.Conv3d(out_channels, out_channels, kernel_size, dilation=dilation),
            spnn.BatchNorm(out_channels),
        )

        if in_channels != out_channels or np.prod(stride) != 1:
            self.shortcut = nn.Sequential(
                spnn.Conv3d(in_channels, out_channels, 1, stride=stride),
                spnn.BatchNorm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = spnn.ReLU(True)

    def forward(self, x: SparseTensor):
        x = self.relu(self.main(x) + self.shortcut(x))
        return x

class SparseEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=256,
        block_channels=[16, 32, 64, 128], # 4 downscale, 512 --> 32 resolution
        layers_per_block=3,
        triplane_resolution=32,
    ):
        super().__init__()

        # resnet backbone
        blocks = []
        cin = in_channels
        for cout in block_channels:
            for i in range(layers_per_block):
                if i == 0:
                    block = SparseConvBlock(cin, cout, kernel_size=3, stride=2)
                else:
                    block = SparseResBlock(cout, cout, kernel_size=3, stride=1)
                blocks.append(block)
            cin = cout
        self.blocks = nn.ModuleList(blocks)

        # to triplane
        self.out0 = spnn.ToDenseBEVConvolution(cin, out_channels, shape=[triplane_resolution] * 3, dim=0)
        self.out1 = spnn.ToDenseBEVConvolution(cin, out_channels, shape=[triplane_resolution] * 3, dim=1)
        self.out2 = spnn.ToDenseBEVConvolution(cin, out_channels, shape=[triplane_resolution] * 3, dim=2)
    
    def forward(self, x: SparseTensor):
        # x: B pointclouds
        # return: triplane of [B, 3, C, H, W]

        for block in self.blocks:
            x = block(x)
        
        y0 = self.out0(x) # [B, C, H, W]
        y1 = self.out1(x)
        y2 = self.out2(x)

        out = torch.stack([y0, y1, y2], dim=1)
        
        return out

if __name__ == "__main__":
    import kiui
    from kiui.nn.utils import count_parameters
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SparseEncoder().to(device)
    print(model)
    
    total, trainable = count_parameters(model)
    print(f'[INFO] param total: {total/1024**2:.2f}M, trainable: {trainable/1024**2:.2f}M')

    # construct batched points
    def get_random_input():
        N = np.random.randint(1024, 4096, dtype=np.int32)
        points_np = np.random.rand(N, 3) * 2 - 1
        voxel_size = 1 / 256 # 512^3 grid for for [-1, 1]^3
        coords, indices = sparse_quantize(points_np, voxel_size, return_index=True)
        points = SparseTensor(coords=torch.from_numpy(coords).int(), feats=torch.from_numpy(points_np[indices]).float())
        return points

    x = sparse_collate([get_random_input() for _ in range(2)]).to(device)
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        y = model(x)
        kiui.lo(y)

        mem_free, mem_total = torch.cuda.mem_get_info()
        print(f'[INFO] mem forward: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G')

        # test backward
        y.sum().backward()
        mem_free, mem_total = torch.cuda.mem_get_info()
        print(f'[INFO] mem backward: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G')