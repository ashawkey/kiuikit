import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Tuple, Literal
from functools import partial
from torch.utils.checkpoint import checkpoint

from kiui.nn.attention_xformers import MemEffAttention, MemEffCrossAttention

class Timesteps(nn.Module):
    def __init__(self, num_channels: int = 320):
        super().__init__()
        self.num_channels = num_channels

        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32)
        exponent = torch.exp(exponent / half_dim)
        self.register_buffer('exponent', exponent)


    def forward(self, timesteps):
        # timesteps: [B,]
        emb = timesteps[:, None].float() * self.exponent[None, :]
        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, out_channels, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_channels, out_channels, bias=True)

    def forward(self, sample):
        # sample: [B, C]
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)

        return sample


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

        # self-attention
        self.ln1 = nn.LayerNorm(channels)
        self.attention = MemEffAttention(dim=channels, num_heads=num_heads, gradient_checkpointing=gradient_checkpointing)

        # cross-attention
        self.ln2 = nn.LayerNorm(channels)
        self.cross_attention = MemEffCrossAttention(dim=channels, dim_q=channels, dim_k=cond_channels, dim_v=cond_channels, num_heads=num_heads, gradient_checkpointing=gradient_checkpointing)

        # MLP
        self.ln3 = nn.LayerNorm(channels)
        self.mlp = FeedForward(in_channels=channels, out_channels=channels)


    def forward(self, x, cond):
        # x: [B, N, channels], embeddings, Q in cross-attention, QKV in self-attention
        # cond: [B, M, cond_channels], features, KV in cross-attention (should be pre-normalized)

        x = x + self.attention(self.ln1(x))
        x = x + self.cross_attention(self.ln2(x), cond, cond)
        x = x + self.mlp(self.ln3(x))

        return x


class ImageTransformer(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        num_layers: int = 2,
        num_heads: int = 16,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual
        self.skip_scale = skip_scale

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.proj_in = nn.Linear(in_channels, out_channels, bias=True)
        self.blocks = nn.ModuleList([
            ResAttBlock(channels=out_channels, cond_channels=cond_channels, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.proj_out = nn.Linear(out_channels, out_channels, bias=True)

    def forward(self, x, cond):
        # x: [B, C, H, W]
        # cond: [B, L, C']

        B, C, H, W = x.shape
        
        res = x
        x = self.norm(x)

        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x, cond)
        x = self.proj_out(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        if self.residual:
            x = (x + res) * self.skip_scale
            
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        resample: Literal['default', 'up', 'down'] = 'default',
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1, # multiplied to output
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels
        self.skip_scale = skip_scale

        self.time_emb_proj = nn.Linear(temb_channels, out_channels, bias=True)

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = F.silu

        self.resample = None
        if resample == 'up':
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        elif resample == 'down':
            self.resample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    
    def forward(self, x, temb):
        res = x

        x = self.norm1(x)
        x = self.act(x)

        if self.resample:
            res = self.resample(res)
            x = self.resample(x)
        
        x = self.conv1(x)

        temb = self.act(temb)
        temb = self.time_emb_proj(temb).unsqueeze(-1).unsqueeze(-1)
        x = x + temb

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x

class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        cond_channels: int,
        num_layers: int = 2,
        num_att_layers: int = 2,
        downsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
 
        nets = []
        attns = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock(cin, out_channels, temb_channels, skip_scale=skip_scale))
            if attention:
                attns.append(ImageTransformer(out_channels, out_channels, cond_channels, num_att_layers, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, temb, cond):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, temb, cond, use_reentrant=False)
        else:
            return self._forward(x, temb, cond)
    
    def _forward(self, x, temb, cond):
        xs = []

        for attn, net in zip(self.attns, self.nets):
            x = net(x, temb)
            if attn:
                x = attn(x, cond)
            xs.append(x)

        if self.downsample:
            x = self.downsample(x)
            xs.append(x)
  
        return x, xs


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        cond_channels: int,
        num_layers: int = 2,
        num_att_layers: int = 2,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        nets = []
        attns = []
        # first layer
        nets.append(ResnetBlock(in_channels, in_channels, temb_channels, skip_scale=skip_scale))
        # more layers
        for i in range(num_layers):
            nets.append(ResnetBlock(in_channels, in_channels, temb_channels, skip_scale=skip_scale))
            if attention:
                attns.append(ImageTransformer(in_channels, in_channels, cond_channels, num_att_layers, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        
    def forward(self, x, temb, cond):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, temb, cond, use_reentrant=False)
        else:
            return self._forward(x, temb, cond)
    
    def _forward(self, x, temb, cond):
        x = self.nets[0](x, temb)
        for attn, net in zip(self.attns, self.nets[1:]):
            if attn:
                x = attn(x, cond)
            x = net(x, temb)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_out_channels: int,
        out_channels: int,
        temb_channels: int,
        cond_channels: int,
        num_layers: int = 2,
        num_att_layers: int = 2,
        upsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        nets = []
        attns = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            cskip = prev_out_channels if (i == num_layers - 1) else out_channels

            nets.append(ResnetBlock(cin + cskip, out_channels, temb_channels, skip_scale=skip_scale))
            if attention:
                attns.append(ImageTransformer(out_channels, out_channels, cond_channels, num_att_layers, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.upsample = None
        if upsample:
            self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, xs, temb, cond):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, xs, temb, cond, use_reentrant=False)
        else:
            return self._forward(x, xs, temb, cond)
    
    def _forward(self, x, xs, temb, cond):

        for attn, net in zip(self.attns, self.nets):
            res_x = xs[-1]
            xs = xs[:-1]
            x = torch.cat([x, res_x], dim=1)
            x = net(x, temb)
            if attn:
                x = attn(x, cond)
            
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            x = self.upsample(x)

        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        temb_channels: int = 512,
        cond_channels: int = 768,
        down_channels: Tuple[int, ...] = (64, 128, 256, 512),
        down_attention: Tuple[bool, ...] = (False, True, True, True),
        mid_attention: bool = True,
        up_channels: Tuple[int, ...] = (512, 256, 128, 64),
        up_attention: Tuple[bool, ...] = (True, True, True, False),
        num_layers: int = 2,
        num_att_layers: int = 2,
        skip_scale: float = np.sqrt(0.5),
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        # time embedding
        self.time_proj = Timesteps(320)
        self.time_embedding = TimestepEmbedding(320, temb_channels)

        # first
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)

        # down
        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(DownBlock(
                cin, cout, temb_channels, cond_channels,
                num_layers=num_layers, 
                num_att_layers=num_att_layers, 
                downsample=(i != len(down_channels) - 1), # not final layer
                attention=down_attention[i],
                skip_scale=skip_scale,
                gradient_checkpointing=gradient_checkpointing,
            ))
        self.down_blocks = nn.ModuleList(down_blocks)

        # mid
        self.mid_block = MidBlock(
            down_channels[-1], temb_channels, cond_channels, 
            num_layers=num_layers, 
            num_att_layers=num_att_layers,
            attention=mid_attention, 
            skip_scale=skip_scale,
            gradient_checkpointing=gradient_checkpointing,
        )

        # up
        up_blocks = []
        cout = up_channels[0]
        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))] # for assymetric

            up_blocks.append(UpBlock(
                cin, cskip, cout, temb_channels, cond_channels,
                num_layers=num_layers + 1, # one more layer for up
                num_att_layers=num_att_layers,
                upsample=(i != len(up_channels) - 1), # not final layer
                attention=up_attention[i],
                skip_scale=skip_scale,
                gradient_checkpointing=gradient_checkpointing,
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # last
        self.norm_out = nn.GroupNorm(num_channels=up_channels[-1], num_groups=32, eps=1e-5)
        self.conv_out = nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x, t, cond):
        # x: [B, C, H, W], sample
        # t: [B,], timestep
        # cond: [B, L, C'], cross-att features

        # time
        temb = self.time_proj(t).to(x.dtype)
        temb = self.time_embedding(temb)

        # first
        x = self.conv_in(x)
        
        # down
        xss = [x]
        for block in self.down_blocks:
            x, xs = block(x, temb, cond)
            xss.extend(xs)
        
        # mid
        x = self.mid_block(x, temb, cond)

        # up
        for block in self.up_blocks:
            xs = xss[-len(block.nets):]
            xss = xss[:-len(block.nets)]
            x = block(x, xs, temb, cond)

        # last
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x) # [B, Cout, H', W']

        return x
    
if __name__ == '__main__':
    import kiui
    from kiui.nn.utils import count_parameters
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(gradient_checkpointing=True).to(device)
    print(model)
    
    total, trainable = count_parameters(model)
    print(f'[INFO] param total: {total/1024**2:.2f}M, trainable: {trainable/1024**2:.2f}M')

    # test forward
    x = torch.randn(1, 4, 64, 64, device=device)
    t = torch.randint(0, 1000, (1,), device=device)
    cond = torch.randn(1, 77, 768, device=device)
    kiui.lo(x, t, cond)
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        y = model(x, t, cond)
        kiui.lo(y)
        
        mem_free, mem_total = torch.cuda.mem_get_info()
        print(f'[INFO] mem forward: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G')

        # test backward
        loss = y.mean()
        loss.backward()
        mem_free, mem_total = torch.cuda.mem_get_info()
        print(f'[INFO] mem backward: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G')    