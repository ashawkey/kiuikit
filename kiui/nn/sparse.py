import math
import numpy as np
from typing import List, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

''' 
# install from source (modified from the original repo to make installation simpler)
sudo apt install libsparsehash-dev
pip install git+https://github.com/ashawkey/torchsparse
'''

import torchsparse
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.utils.collate import sparse_collate, sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

class ToDenseBEVConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shape: Union[List[int], Tuple[int, int, int], torch.Tensor],
        offset: Tuple[int, int, int] = (0, 0, 0),
        dim: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer("offset", torch.IntTensor([list(offset) + [0]]))
        if isinstance(shape, torch.Tensor):
            self.register_buffer("shape", shape.int())
        else:
            self.register_buffer("shape", torch.IntTensor(shape))
        self.dim = dim
        self.n_kernels = int(self.shape[self.dim])
        self.bev_dims = [i for i in range(3) if i != self.dim]
        self.bev_shape = self.shape[self.bev_dims]
        self.kernel = nn.Parameter(torch.zeros(self.n_kernels, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(1, out_channels)) if bias else 0
        self.reset_parameters()

    def extra_repr(self):
        return "in_channels={}, out_channels={}, n_kernels={}".format(
            self.in_channels, self.out_channels, self.n_kernels
        )

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.in_channels)
        self.kernel.data.uniform_(-std, std)

    def forward(self, input: SparseTensor) -> torch.Tensor:
        coords, feats, stride = input.coords, input.feats, input.stride
        stride = torch.tensor(stride).unsqueeze(dim=0).to(feats)[:, self.dim]

        kernel = torch.index_select(self.kernel, 0, torch.div(coords[:, self.dim], stride).trunc().long())
        feats = (feats.unsqueeze(dim=-1) * kernel).sum(1) + self.bias
        coords = (coords - self.offset).t()[[0] + self.bev_dims].long() # fix ref: https://github.com/mit-han-lab/torchsparse/issues/296
        coords[1:] = torch.div(coords[1:], stride).trunc().long()
        indices = (
            coords[0] * int(self.bev_shape.prod())
            + coords[1] * int(self.bev_shape[1])
            + coords[2]
        )
        batch_size = coords[0].max().item() + 1
        output = torch.sparse_coo_tensor(
            indices.unsqueeze(dim=0),
            feats,
            torch.Size([batch_size * int(self.bev_shape.prod()), feats.size(-1)]),
        ).to_dense()
        output = output.view(batch_size, *self.bev_shape, -1)
        output = output.permute(0, 3, 1, 2).contiguous()
        return output

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
        block_channels=[32, 64, 128, 256], # 4 downscale, 512 --> 32 resolution
        layers_per_block=3,
        resolution=32,
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
        self.out0 = ToDenseBEVConvolution(cin, out_channels, shape=[resolution] * 3, dim=0)
        self.out1 = ToDenseBEVConvolution(cin, out_channels, shape=[resolution] * 3, dim=1)
        self.out2 = ToDenseBEVConvolution(cin, out_channels, shape=[resolution] * 3, dim=2)
    
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
    # print(model)
    
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
    # kiui.lo(x)
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        y = model(x)
        kiui.lo(y)

        mem_free, mem_total = torch.cuda.mem_get_info()
        print(f'[INFO] mem forward: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G')

        # test backward
        y.sum().backward()
        mem_free, mem_total = torch.cuda.mem_get_info()
        print(f'[INFO] mem backward: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G')
