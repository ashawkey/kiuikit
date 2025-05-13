from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from kiui.nn.attention_flash import attention

@dataclass
class Options:
    vocab_size: int = 25600
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_layers: int = 32
    max_position_embeddings: int = 2048 # context length, also rope cache length
    num_attention_heads: int = 16
    attn_dropout: float = 0.0
    attn_bias: bool = False
    mlp_bias: bool = False
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000
    initializer_range: float = 0.02
    use_gradient_checkpointing: bool = True # only at training
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim, # head dim of attention
        max_position_embeddings=2048,
        theta=500000,
    ):
        super().__init__()
        self.max_seq_len_cached = max_position_embeddings
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids):
        # position_ids: [B, N]

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1) # [B, C/2, 1]
        position_ids_expanded = position_ids[:, None, :].float() # [B, 1, N]

        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = position_ids.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2) # [B, N, C/2]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        
        return cos, sin # [B, N, C], [B, N, C]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, N, H, C]
    # sin, cos: [B, N, C]
    cos = cos.unsqueeze(2).to(q.dtype)
    sin = sin.unsqueeze(2).to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MLP(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()
        self.opt = opt
        self.hidden_size = opt.hidden_size
        self.intermediate_size = opt.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=opt.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=opt.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=opt.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, x):        
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):

    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()
        self.opt = opt

        self.embed_dim = opt.hidden_size
        self.num_heads = opt.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads # assert divisible

        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=opt.attn_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=opt.attn_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=opt.attn_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=opt.attn_bias)

    def forward(
        self,
        hidden_states: torch.Tensor, # [B, N, C]
        pos_embeds: Tuple[torch.Tensor], # [B, N, C] x 2
        cache_kv: Optional[Tuple[torch.Tensor]] = None, # (key, value), [B, N', h, c]
        attention_mask: Optional[torch.Tensor] = None, # [B, N]
    ):

        B, N, C = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling # [B, N, C]
        query_states = query_states.view(B, N, self.num_heads, self.head_dim)
       
        # get key & value proj
        key_states = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)

        # apply rope
        cos, sin = pos_embeds
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # may concat cached key & value
        if cache_kv is not None:
            key_states = torch.cat([cache_kv[0], key_states], dim=1)
            value_states = torch.cat([cache_kv[1], value_states], dim=1)

        # update cache
        cache_kv = (key_states, value_states)
     
        # self-attention
        attn_output = attention(query_states, key_states, value_states, mask_q=attention_mask, mask_kv=attention_mask, dropout=self.opt.attn_dropout if self.training else 0.0, causal=True)
        attn_output = attn_output.reshape(B, N, self.num_heads * self.head_dim)

        # output proj
        attn_output = self.out_proj(attn_output)

        return attn_output, cache_kv


class DecoderLayer(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()
        self.opt = opt
        self.self_attn = Attention(opt=opt)
        self.mlp = MLP(opt)
        self.input_layernorm = RMSNorm(opt.hidden_size, eps=opt.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(opt.hidden_size, eps=opt.rms_norm_eps)

    def forward(self, hidden_states, pos_embeds, attention_mask=None, cache_kv=None):
        if self.training and self.opt.use_gradient_checkpointing:
            return checkpoint(self._forward, hidden_states, pos_embeds, attention_mask, cache_kv, use_reentrant=False)
        else:
            return self._forward(hidden_states, pos_embeds, attention_mask, cache_kv)

    def _forward(
        self,
        hidden_states: torch.Tensor, # [B, N, C]
        pos_embeds: Tuple[torch.Tensor], # [B, N, C] x 2
        attention_mask: Optional[torch.Tensor] = None, # [B, N]
        cache_kv: Optional[Tuple[torch.Tensor]] = None, # cached (key, value)
    ):
      
        residual = hidden_states
 
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, new_cache_kv = self.self_attn(
            hidden_states=hidden_states,
            pos_embeds=pos_embeds,
            cache_kv=cache_kv,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_cache_kv


class Model(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

        self.embed_tokens = nn.Embedding(opt.vocab_size, opt.hidden_size, padding_idx=opt.pad_token_id)
        self.layers = nn.ModuleList([DecoderLayer(opt) for _ in range(opt.num_layers)])
        self.norm = RMSNorm(opt.hidden_size, eps=opt.rms_norm_eps)
        self.rope = RotaryEmbedding(opt.hidden_size // opt.num_attention_heads, max_position_embeddings=opt.max_position_embeddings, theta=opt.rope_theta)

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.opt.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor = None, # [B, N]
        inputs_embeds: Optional[torch.Tensor] = None, # [B, N, C]
        attention_mask: Optional[torch.Tensor] = None, # [B, N]
        all_cache_kv: Optional[List[torch.Tensor]] = None, # list of cache_kv for all layers
        use_cache: bool = False,
    ):
       
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            B, N = input_ids.shape
        elif inputs_embeds is not None:
            B, N, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # get current length
        past_length = all_cache_kv[0][0].shape[1] if all_cache_kv is not None else 0
        mask_seq_length = past_length + N
        position_ids = torch.arange(past_length, mask_seq_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)

        # attention mask
        attention_mask = torch.ones(B, mask_seq_length, device=inputs_embeds.device) if attention_mask is None else attention_mask
        
        # 2d mask for flash-attn (if all 1, no need for mask)
        causal_attention_mask = attention_mask if (0 in attention_mask) else None
        
        hidden_states = inputs_embeds

        # position embeddings
        pos_embeds = self.rope(position_ids)

        # decoder layers
        new_all_cache_kv = [] if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):

            cache_kv = all_cache_kv[idx] if all_cache_kv is not None else None

            hidden_states, new_cache_kv = decoder_layer(hidden_states, pos_embeds, attention_mask=causal_attention_mask, cache_kv=cache_kv)

            # update cache if inference
            if new_all_cache_kv is not None:
                new_all_cache_kv.append(new_cache_kv)

        hidden_states = self.norm(hidden_states)

        return hidden_states, new_all_cache_kv

class CausalModel(nn.Module):
  
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt
        self.model = Model(opt)
        self.lm_head = nn.Linear(opt.hidden_size, opt.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None, # [B, N]
        inputs_embeds: Optional[torch.FloatTensor] = None, # [B, M, C]
        attention_mask: Optional[torch.Tensor] = None, # [B, N]
        all_cache_kv: Optional[List[torch.FloatTensor]] = None, # list of cache_kv for all layers
        labels: Optional[torch.LongTensor] = None, # [B, N]
        use_cache: bool = False,
    ):
       
        hidden_states, new_all_cache_kv = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            all_cache_kv=all_cache_kv,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states).contiguous()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.opt.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels)

        return loss, logits, new_all_cache_kv

    # NOTE: 
    # - just implement the sample method
    # - assume always use cache 
    # - only single batch
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None, # [1, N] or None (will init as BOS [1, 1])
        inputs_embeds: Optional[torch.FloatTensor] = None, # [1, M, C]
        max_length: int = 1000,
        sample: bool = False,
        use_cache: bool = True,
    ):
        
        all_cache_kv = None

        # loop until EOS or max_length
        while True:

            # prepare inputs for this iteration
            model_inputs = {}
            model_inputs['use_cache'] = use_cache
            if all_cache_kv is None: 
                # first iteration may need inputs_embeds
                if inputs_embeds is not None:
                    model_inputs["inputs_embeds"] = inputs_embeds
                else:
                    # may initialize input_ids
                    if input_ids is None:
                        input_ids = torch.full((1, 1), self.opt.bos_token_id, dtype=torch.long, device=self.model.embed_tokens.weight.device)
                    model_inputs["input_ids"] = input_ids
            else:
                # has cache, only need last token
                model_inputs["input_ids"] = input_ids[:, -1:]
                model_inputs["all_cache_kv"] = all_cache_kv

            hidden_states, new_all_cache_kv = self.model(**model_inputs)

            logits = self.lm_head(hidden_states) # [B, 1, V]
            next_token_logits = logits[:, -1, :] # [B, V]

            # TODO: impl constraints here

            # token selection
            if sample:
                probs = F.softmax(next_token_logits, dim=-1)
                # TODO: impl topk/topp sampling here
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1) # [B]
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1) # [B]

            # update input_ids and cache
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=-1)
            all_cache_kv = new_all_cache_kv

            if next_tokens.item() == self.opt.eos_token_id or input_ids.shape[1] >= max_length:
                break
       
        return input_ids


if __name__ == '__main__':

    import time
    import kiui
    from kiui.nn.utils import count_parameters

    kiui.seed_everything(42)

    opt = Options()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalModel(opt).to(device)
    # print(model)
    
    total, trainable = count_parameters(model)
    print(f'[INFO] param total: {total/1024**2:.2f}M, trainable: {trainable/1024**2:.2f}M')

    # test batch forward
    seqs = torch.randint(0, opt.vocab_size, (2, 1024), device=device)
    input_ids = seqs[:, :-1]
    labels = seqs[:, 1:]
    masks = torch.ones_like(input_ids)
    masks[0, 512:] = 0 # mask half of the first sequence

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        loss, logits, _ = model(input_ids=input_ids, attention_mask=masks, labels=labels)
        kiui.lo(loss, logits)
        
        mem_free, mem_total = torch.cuda.mem_get_info()
        print(f'[INFO] mem forward: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G')

        # test backward
        loss.backward()
        mem_free, mem_total = torch.cuda.mem_get_info()
        print(f'[INFO] mem backward: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G')  

    # test generate
    model.eval()
    for _ in range(3):
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                t0 = time.time()
                seqs = model.generate(sample=False, max_length=256, use_cache=True) # must be sufficient long to show the acceleration of use_cache
                torch.cuda.synchronize()
                t1 = time.time()
                print(f'[INFO] generate time: {t1-t0:.2f}s')
                mem_free, mem_total = torch.cuda.mem_get_info()
                print(f'[INFO] mem generate: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G')  