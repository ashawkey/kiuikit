import sys
sys.path.append('.')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Tuple, Literal
from functools import partial
from torch.utils.checkpoint import checkpoint

from kiui.nn.attention_xformers import MemEffAttention, MemEffCrossAttention

class GEGLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels * 2, bias=True)

    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return x1 * F.gelu(x2)


class FeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.ModuleList([
            GEGLU(in_channels, out_channels * 4),    
            nn.Linear(out_channels * 4, out_channels, bias=True),
        ])

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x
    

class ResAttBlock(nn.Module):
    def __init__(self, channels=1024, cond_channels=768, num_heads=16, gradient_checkpointing=False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        # self-attention
        self.ln1 = nn.LayerNorm(channels)
        self.attention = MemEffCrossAttention(dim=channels, dim_q=channels, dim_k=cond_channels, dim_v=cond_channels, num_heads=num_heads)

        # MLP
        self.ln2 = nn.LayerNorm(channels)
        self.mlp = FeedForward(in_channels=channels, out_channels=channels)


    def forward(self, x, cond):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, cond, use_reentrant=False)
        else:
            return self._forward(x, cond)
    
    def _forward(self, x, cond):
        # x: [B, N, channels], embeddings, Q in cross-attention, QKV in self-attention
        # cond: [B, M, cond_channels], features, KV in cross-attention (should be pre-normalized)

        x = x + self.attention(self.ln1(x), cond, cond)
        x = x + self.mlp(self.ln2(x))

        return x


class Perciever(nn.Module):
    def __init__(
        self,
        embed_size: int = 77,
        embed_channels: int = 1024,
        cond_channels: int = 1280,
        channels: int = 768,
        num_layers: int = 4,
        num_heads: int = 16,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.embeds = nn.Parameter(torch.randn(1, embed_size, channels) / channels ** 0.5)

        self.cond_proj = nn.Linear(cond_channels, channels)
        self.cond_ln = nn.LayerNorm(channels)

        self.blocks = nn.ModuleList([
            ResAttBlock(channels=channels, cond_channels=channels, num_heads=num_heads, gradient_checkpointing=gradient_checkpointing)
            for _ in range(num_layers)
        ])

        self.embed_proj = nn.Linear(channels, embed_channels)
        self.embed_ln = nn.LayerNorm(embed_channels)
    
    def forward(self, cond):
        # cond: [N, L, C], cond features

        x = self.embeds.repeat(cond.shape[0], 1, 1)

        cond = self.cond_ln(self.cond_proj(cond))

        for block in self.blocks:
            x = block(x, cond)
        
        x = self.embed_ln(self.embed_proj(x))
        
        return x

    
if __name__ == '__main__':
    import kiui
    from kiui.nn.utils import count_parameters
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Perciever(gradient_checkpointing=True).to(device)
    print(model)
    
    total, trainable = count_parameters(model)
    print(f'[INFO] param total: {total/1024**2:.2f}M, trainable: {trainable/1024**2:.2f}M')

    # test forward
    cond = torch.randn(1, 257, 1280, device=device)
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        y = model(cond)
        kiui.lo(y)
        
        mem_free, mem_total = torch.cuda.mem_get_info()
        print(f'[INFO] mem forward: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G')

        # test backward
        loss = y.mean()
        loss.backward()
        mem_free, mem_total = torch.cuda.mem_get_info()
        print(f'[INFO] mem backward: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G')    