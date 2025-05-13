import torch
import torch.nn as nn
import torch.nn.functional as F

import kiui

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    FLASH_ATTN_AVAILABLE = True
except:
    print('[WARN] flash_attn not available, using naive implementation')
    FLASH_ATTN_AVAILABLE = False

def attention(q, k, v, mask_q=None, mask_kv=None, dropout=0, causal=False):
    # q: (B, N, H, D)
    # k: (B, M, H, D)
    # v: (B, M, H, D)
    # mask_q: (B, N)
    # mask_kv: (B, M)
    # return: (B, N, H, D)

    B, N, H, D = q.shape
    M = k.shape[1]

    # kiui.lo(q, k, v)

    if causal: 
        assert N == 1 or N == M, 'Causal mask only supports self-attention'

    # unmasked case
    if mask_q is None and mask_kv is None:
        if FLASH_ATTN_AVAILABLE:
            return flash_attn_func(q, k, v, dropout, causal=causal) # [B, N, H, D]
        else:
            # naive implementation
            q = q.transpose(1, 2).reshape(B * H, N, D)
            k = k.transpose(1, 2).reshape(B * H, M, D)
            v = v.transpose(1, 2).reshape(B * H, M, D)
            w = torch.bmm(q, k.transpose(1, 2)) / (D ** 0.5) # [B*H, N, M]
            if causal and N > 1:
                causal_mask = torch.full((N, M), float('-inf'), device=w.device, dtype=w.dtype)
                causal_mask = torch.triu(causal_mask, diagonal=1)
                w = w + causal_mask.unsqueeze(0)
            w = F.softmax(w, dim=-1)
            if dropout > 0:
                w = F.dropout(w, p=dropout)
            out = torch.bmm(w, v) # [B*H, N, D]
            out = out.reshape(B, H, N, D).transpose(1, 2).contiguous() # [B, N, H, D]
            return out
    
    # at least one of q or kv is masked, but we need both masks for varlen_func anyway
    if mask_q is None:
        mask_q = torch.ones(B, N, dtype=torch.bool, device=q.device)
    elif mask_kv is None:
        mask_kv = torch.ones(B, M, dtype=torch.bool, device=q.device)

    if FLASH_ATTN_AVAILABLE:
        # unpad (gather) input
        # mask_q: [B, N], first row has N1 1s, second row has N2 1s, ...
        # indices: [Ns,], Ns = N1 + N2 + ...
        # cu_seqlens_q: [B+1,], (0, N1, N1+N2, ...), cu=cumulative
        # max_len_q: scalar, max(N1, N2, ...)
        q, indices_q, cu_seqlens_q, max_len_q = unpad_input(q, mask_q)
        k, indices_kv, cu_seqlens_kv, max_len_kv = unpad_input(k, mask_kv)
        v = index_first_axis(v.reshape(-1, H, D), indices_kv) # same indice as k

        # call varlen_func
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_len_q,
            max_seqlen_k=max_len_kv,
            dropout_p=dropout,
            causal=causal,
        )

        # pad (put back) output
        out = pad_input(out, indices_q, B, N)
        return out
    else:
        raise NotImplementedError('masked attention requires flash_attn!')


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, input_dim=None, output_dim=None, dropout=0, causal=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, 'hidden_dim must be divisible by num_heads'
        self.head_dim = hidden_dim // num_heads
        self.causal = causal
        self.dropout = dropout

        self.qkv_proj = nn.Linear(self.input_dim, 3 * self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x, mask=None):
        # x: [B, N, C]
        # mask: [B, N]
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.chunk(3, dim=0) # [3, B, N, H, D] -> 3 * [1, B, N, H, D]
        x = attention(q[0], k[0], v[0], mask_q=mask, mask_kv=mask, dropout=self.dropout, causal=self.causal) # [B, N, H, D]
        x = self.out_proj(x.reshape(B, N, -1))
        return x


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, input_dim=None, context_dim=None, output_dim=None, dropout=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        self.context_dim = context_dim if context_dim is not None else hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, 'hidden_dim must be divisible by num_heads'
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.context_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.context_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x, context, mask_q=None, mask_kv=None):
        # x: [B, N, C]
        # context: [B, M, C']
        # mask_q: [B, N]
        # mask_kv: [B, M]
        B, N, C = x.shape
        M = context.shape[1]
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(context).reshape(B, M, self.num_heads, self.head_dim)
        v = self.v_proj(context).reshape(B, M, self.num_heads, self.head_dim)
        x = attention(q, k, v, mask_q=mask_q, mask_kv=mask_kv, dropout=self.dropout, causal=False) # [B, N, H, D]
        x = self.out_proj(x.reshape(B, N, -1))
        return x