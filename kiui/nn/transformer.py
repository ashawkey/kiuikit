import torch
import torch.nn as nn
import torch.nn.functional as F

from kiui.typing import *
from kiui.nn.attention import MemEffAttention, MemEffCrossAttention

# the usually used 2-layer MLP in transformer
class FeedForward(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or 4 * in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    

# a typical transformer block.
# disable self-attention and it's a perciever block.
class ResAttBlock(nn.Module):
    def __init__(self, dim=1024, dim_e=768, num_heads=16, self_att=True, gradient_checkpointing=False):
        super().__init__()

        # cross-attention
        self.ln1 = nn.LayerNorm(dim)
        self.cross_attention = MemEffCrossAttention(dim=dim, dim_q=dim, dim_k=dim_e, dim_v=dim_e, num_heads=num_heads, gradient_checkpointing=gradient_checkpointing)

        # self-attention
        self.self_att = self_att
        if self_att:
            self.ln2 = nn.LayerNorm(dim)
            self.attention = MemEffAttention(dim=dim, num_heads=num_heads, gradient_checkpointing=gradient_checkpointing)

        # MLP
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = FeedForward(in_features=dim, hidden_features=dim * 4, out_features=dim)


    def forward(self, x, f):
        # x: [B, N, dim], embeddings, Q in cross-attention, QKV in self-attention
        # f: [B, M, dim_e], features, KV in cross-attention (should be pre-normalized)

        x = x + self.cross_attention(self.ln1(x), f, f) # [B, N, dim]
        
        if self.self_att:
            x = x + self.attention(self.ln2(x)) # [B, N, dim]

        x = x + self.mlp(self.ln3(x)) # [B, N, dim]

        return x

# RMSNorm used in llama in place of LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor):
        # x: [B, N, dim]
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(dtype)

class LlamaMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()

        self.gate_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.down_proj = nn.Linear(hidden_features, in_features, bias=False)
        self.act = F.silu

    def forward(self, x):
        # x: [..., C]
        x = self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        return x