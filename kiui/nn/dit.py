import sys
sys.path.append('.')
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np

from kiui.nn.attention_flash import SelfAttention, CrossAttention

import kiui

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Timesteps(nn.Module):
    def __init__(self, num_channels=256, flip_sin_to_cos=False, downscale_freq_shift=0, scale=1, max_period=10000):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        # scale embeddings
        emb = self.scale * emb

        # concat sine and cosine embeddings
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # flip sine and cosine embeddings
        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        # zero pad
        if self.num_channels % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
            
        return emb
    

class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        self.act = F.silu
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, sample_proj_bias)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


# PixArtAlpha-style
class DiTLayer(nn.Module):
    def __init__(self, dim, num_heads, gradient_checkpointing=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing

        self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn1 = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn2 = CrossAttention(dim, num_heads)
        self.ff = FeedForward(dim)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim ** 0.5)


    def forward(self, x, c, t_adaln):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, c, t_adaln, use_reentrant=False)
        else:
            return self._forward(x, c, t_adaln)
    
    def _forward(self, x, c, t_adaln):
        # x: [B, N, C], hidden states
        # c: [B, M, C], condition (assume normed and projected to C)
        # t_adaln: [B, 6, C], timestep embedding of adaln
        # return: [B, N, C], updated hidden states

        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t_adaln).chunk(6, dim=1)

        x = self.norm1(x)
        x = x * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn1(x)
        
        x = x + self.attn2(x, c)

        x = self.norm2(x)
        x = x * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.ff(x)

        return x


class DiT(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=16, latent_size=2048, latent_dim=64, num_layers=24, gradient_checkpointing=True):
        super().__init__()

        # project in
        self.proj_in = nn.Linear(latent_dim, hidden_dim)

        # positional encoding (just use a learnable positional encoding)
        self.pos_embed = nn.Parameter(torch.randn(1, latent_size, hidden_dim) / hidden_dim ** 0.5)

        # timestep encoding
        self.timestep_embed = Timesteps(num_channels=256)
        self.timestep_proj = TimestepEmbedding(256, hidden_dim)
        self.adaln_linear = nn.Linear(hidden_dim, hidden_dim * 6, bias=True)

        # transformer layers
        self.layers = nn.ModuleList([DiTLayer(hidden_dim, num_heads, gradient_checkpointing) for _ in range(num_layers)])

        # project out
        self.norm_out = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_dim) / hidden_dim ** 0.5)
        self.proj_out = nn.Linear(hidden_dim, latent_dim)

       
    def forward(self, x, c, t):
        # x: [B, N, C], hidden states
        # c: [B, M, C], condition (assume normed and projected to C)
        # t: [B,], timestep
        # return: [B, N, C], updated hidden states

        B, N, C = x.shape

        # project in
        x = self.proj_in(x)

        # positional encoding
        x = x + self.pos_embed

        # timestep encoding
        t_emb = self.timestep_embed(t)
        t_emb = self.timestep_proj(t_emb) # [B, C]
        t_adaln = self.adaln_linear(F.silu(t_emb)).view(B, 6, -1) # [B, 6, C]

        # transformer layers
        for layer in self.layers:
            x = layer(x, c, t_adaln)
        
        # project out
        shift, scale = (self.scale_shift_table[None] + t_emb[:, None]).chunk(2, dim=1)
        x = self.norm_out(x)
        x = x * (1 + scale) + shift
        x = self.proj_out(x)

        return x
    

if __name__ == '__main__':
    import kiui
    from kiui.nn.utils import count_parameters
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiT(gradient_checkpointing=True).to(device)
    print(model)
    
    total, trainable = count_parameters(model)
    print(f'[INFO] param total: {total/1024**2:.2f}M, trainable: {trainable/1024**2:.2f}M')

    # test forward
    x = torch.randn(4, 2048, 64, device=device, dtype=torch.float16)
    t = torch.randint(0, 1000, (4,), device=device)
    c = torch.randn(4, 257, 1024, device=device, dtype=torch.float16)
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        y = model(x, c, t)
        kiui.lo(y)
        
        mem_free, mem_total = torch.cuda.mem_get_info()
        print(f'[INFO] mem forward: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G')

        # test backward
        loss = y.mean()
        loss.backward()
        mem_free, mem_total = torch.cuda.mem_get_info()
        print(f'[INFO] mem backward: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G')