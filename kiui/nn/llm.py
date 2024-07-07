from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

@dataclass
class Options:
    vocab_size: int = 25600
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_layers: int = 32
    max_position_embeddings: int = 2048
    num_attention_heads: int = 16
    attn_dropout: float = 0.0
    attn_bias: bool = False
    mlp_bias: bool = False
    mlp_act: str = "silu"
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_gradient_checkpointing: bool = True # only at training
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    

class OPTAttention(nn.Module):
    
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
        hidden_states: torch.Tensor, # [B, N, C], query
        cache_kv: Optional[Tuple[torch.Tensor]] = None, # (key, value), [B, N', h, c]
        attention_mask: Optional[torch.Tensor] = None, # [B, 1, N, M], mask (M = N + N')
    ):

        B, N, C = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim) # [B, N, num_heads, head_dim]
        query_states = query_states * self.scaling

        # get key & value proj
        key_states = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim) # [B, N, num_heads, head_dim]
        value_states = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim) # [B, N, num_heads, head_dim]

        # may concat cached key & value
        if cache_kv is not None:
            key_states = torch.cat([cache_kv[0], key_states], dim=1) # [B, M, num_heads, head_dim], M may be larger than N!
            value_states = torch.cat([cache_kv[1], value_states], dim=1) # [B, M, num_heads, head_dim]

        # update cache
        cache_kv = (key_states, value_states)
        M = key_states.shape[1]

        # self-attention
        query_states = query_states.transpose(1, 2).contiguous().view(B * self.num_heads, N, self.head_dim)
        key_states = key_states.transpose(1, 2).contiguous().view(B * self.num_heads, M, self.head_dim)
        value_states = value_states.transpose(1, 2).contiguous().view(B * self.num_heads, M, self.head_dim)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) # [B * num_heads, N, M]

        if attention_mask is not None:
            attn_weights = attn_weights.view(B, self.num_heads, N, M) + attention_mask # [B, num_heads, N, M]
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
            attn_weights = attn_weights.view(B * self.num_heads, N, M) # [B * num_heads, N, M]

        # softmax
        if attn_weights.dtype == torch.float16:
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = F.softmax(attn_weights, dim=-1)

        # may apply dropout
        attn_probs = F.dropout(attn_weights, p=self.opt.attn_dropout, training=self.training)

        # weighted sum
        attn_output = torch.bmm(attn_probs, value_states) # [B * num_heads, N, head_dim]
        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2) # [B, N, num_heads, head_dim]
        attn_output = attn_output.reshape(B, N, self.embed_dim) # [B, N, C]

        # output proj
        attn_output = self.out_proj(attn_output)

        return attn_output, cache_kv


class OptFlashAttention2(OPTAttention):

    def forward(
        self,
        hidden_states: torch.Tensor, # [B, N, C]
        cache_kv: Optional[Tuple[torch.Tensor]] = None, # (key, value), [B, N', h, c]
        attention_mask: Optional[torch.Tensor] = None, # [B, N], note this is different from OPTAttention (which is processed 4D mask [B, 1, N, N])
    ):

        B, N, C = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling # [B, N, C]
        query_states = query_states.view(B, N, self.num_heads, self.head_dim) # [B, N, num_heads, head_dim]
       
        # get key & value proj
        key_states = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)

        # may concat cached key & value
        if cache_kv is not None:
            key_states = torch.cat([cache_kv[0], key_states], dim=1)
            value_states = torch.cat([cache_kv[1], value_states], dim=1)

        # update cache
        cache_kv = (key_states, value_states)
     
        # self-attention
        attn_output = self._flash_attention_forward(query_states, key_states, value_states, attention_mask, dropout=self.opt.attn_dropout if self.training else 0.0)
        attn_output = attn_output.reshape(B, N, self.num_heads * self.head_dim)

        # output proj
        attn_output = self.out_proj(attn_output)

        return attn_output, cache_kv


    def _flash_attention_forward(self, query_states, key_states, value_states, attention_mask, dropout=0.0, softmax_scale=None):
        
        N = query_states.shape[1]

        if attention_mask is not None:
            
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32) # [B,] (N1, N2, ...)
            max_seqlen = seqlens.max().item()
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten() # [M,]
            seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)) # [B+1,] (0, N1, N1+N2, ...) 

            B, M, num_heads, head_dim = key_states.shape
            key_states = index_first_axis(key_states.reshape(B * M, num_heads, head_dim), indices)
            value_states = index_first_axis(value_states.reshape(B * M, num_heads, head_dim), indices)

            if N == M:
                query_states = index_first_axis(query_states.reshape(B * N, num_heads, head_dim), indices)
                
                seqlens_q = seqlens
                max_seqlen_q = max_seqlen

            elif N == 1:
                seqlens_q = torch.arange(B + 1, dtype=torch.int32, device=query_states.device)
                max_seqlen_q = 1

                indices = seqlens_q[:-1]
                query_states = query_states.squeeze(1)

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=seqlens_q,
                cu_seqlens_k=seqlens,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=True,
            )
            attn_output = pad_input(attn_output_unpad, indices, B, N)
        else:
            attn_output = flash_attn_func(query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=True)

        return attn_output

class OPTMLP(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()
        self.opt = opt
        self.hidden_size = opt.hidden_size
        self.intermediate_size = opt.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=opt.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=opt.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=opt.mlp_bias)
        self.act_fn = ACT2FN[opt.mlp_act]

    def forward(self, x):        
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class OPTRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class OPTDecoderLayer(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()
        self.opt = opt
        self.self_attn = OptFlashAttention2(opt=opt)
        self.mlp = OPTMLP(opt)
        self.input_layernorm = OPTRMSNorm(opt.hidden_size, eps=opt.rms_norm_eps)
        self.post_attention_layernorm = OPTRMSNorm(opt.hidden_size, eps=opt.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, cache_kv=None):
        if self.training and self.opt.use_gradient_checkpointing:
            return checkpoint(self._forward, hidden_states, attention_mask, cache_kv, use_reentrant=False)
        else:
            return self._forward(hidden_states, attention_mask, cache_kv)

    def _forward(
        self,
        hidden_states: torch.Tensor, # [B, N, C]
        attention_mask: Optional[torch.Tensor] = None, # [B, N]
        cache_kv: Optional[Tuple[torch.Tensor]] = None, # cached (key, value)
    ):
      
        residual = hidden_states
 
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, new_cache_kv = self.self_attn(
            hidden_states=hidden_states,
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


class OPTModel(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

        self.embed_tokens = nn.Embedding(opt.vocab_size, opt.hidden_size, padding_idx=opt.pad_token_id)
        self.embed_positions = nn.Embedding(opt.max_position_embeddings, opt.hidden_size)

        self.layers = nn.ModuleList([OPTDecoderLayer(opt) for _ in range(opt.num_layers)])
        
        self.norm = OPTRMSNorm(opt.hidden_size, eps=opt.rms_norm_eps)

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
        position_ids = torch.arange(past_length, mask_seq_length, dtype=torch.long, device=inputs_embeds.device)

        # kiui.lo(position_ids)

        # attention mask
        attention_mask = torch.ones(B, mask_seq_length, device=inputs_embeds.device) if attention_mask is None else attention_mask
        
        # 2d mask for flash-attn (if all 1, no need for mask)
        causal_attention_mask = attention_mask if (0 in attention_mask) else None
        
        # position embeddings
        pos_embeds = self.embed_positions(position_ids)
        hidden_states = inputs_embeds + pos_embeds

        # decoder layers
        new_all_cache_kv = [] if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):

            cache_kv = all_cache_kv[idx] if all_cache_kv is not None else None

            hidden_states, new_cache_kv = decoder_layer(hidden_states, attention_mask=causal_attention_mask, cache_kv=cache_kv)

            # update cache if inference
            if new_all_cache_kv is not None:
                new_all_cache_kv.append(new_cache_kv)

        hidden_states = self.norm(hidden_states)

        return hidden_states, new_all_cache_kv

class OPTForCausalLM(nn.Module):
  
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt
        self.model = OPTModel(opt)
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
    model = OPTForCausalLM(opt).to(device)
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
                seqs = model.generate(sample=False, max_length=2048, use_cache=True) # must be sufficient long to show the acceleration of use_cache
                torch.cuda.synchronize()
                t1 = time.time()
                print(f'[INFO] generate time: {t1-t0:.2f}s')
                mem_free, mem_total = torch.cuda.mem_get_info()
                print(f'[INFO] mem generate: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G')  