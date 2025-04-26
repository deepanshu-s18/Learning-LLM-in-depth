# model.py
# FINAL VERSION
# This file defines the complete architecture for a mini-DeepSeek V3 model,
# integrating MLA, Decoupled RoPE, MoE, MTP, and a correct KV Cache with RoPE position offsetting.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Configuration Dataclass ---

@dataclass
class ModelArgs:
    # Architecture
    d_model: int = 512
    n_layers: int = 8
    vocab_size: int = 50257 # Placeholder for tiktoken 'gpt2'
    # Attention (MLA)
    num_heads: int = 8
    d_latent: int = 128
    d_rope: int = 32
    # MoE
    moe_n_routed_experts: int = 8
    moe_n_shared_experts: int = 1
    moe_top_k: int = 2
    moe_routed_hidden: int = 512
    # MTP
    n_mtp_modules: int = 1
    # General
    dropout: float = 0.1
    max_seq_len: int = 1024


# --- Rotary Positional Encoding (RoPE) Helper Module ---

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_head = d_head
        theta = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer('theta', theta)
        
        positions = torch.arange(max_seq_len)
        freqs = torch.outer(positions, self.theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)

    ## NEW ##: Added position_offset for cached inference
    def forward(self, x: torch.Tensor, position_offset: int = 0):
        # x: [B, H, S, D_head]
        seq_len = x.shape[2]
        
        # x_complex: [B, H, S, D_head/2]
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        
        # Get precomputed frequencies using the offset
        # freqs_cis: [S, D_head/2] -> [1, 1, S, D_head/2]
        freqs_cis_slice = self.freqs_cis[position_offset : position_offset + seq_len]
        freqs_cis = freqs_cis_slice.unsqueeze(0).unsqueeze(0)
        
        # Apply rotation via element-wise complex multiplication
        x_rotated = x_complex * freqs_cis
        
        # Cast back to real and reshape
        x_out = torch.view_as_real(x_rotated).flatten(3)
        return x_out.type_as(x)


# --- Multi-Head Latent Attention (MLA) Module ---

class DeepSeekAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model
        self.num_heads = args.num_heads
        self.d_head = args.d_model // args.num_heads
        self.d_latent = args.d_latent
        self.d_rope = args.d_rope

        # Content Path
        self.W_q_content = nn.Linear(args.d_model, args.d_model, bias=False)
        self.W_dkv_content = nn.Linear(args.d_model, args.d_latent, bias=False)
        self.W_uk_content = nn.Linear(args.d_latent, args.d_model, bias=False)
        self.W_uv_content = nn.Linear(args.d_latent, args.d_model, bias=False)

        # Position Path
        self.W_k_pos = nn.Linear(args.d_model, args.d_rope * args.num_heads, bias=False)
        self.W_q_pos = nn.Linear(args.d_model, args.d_rope * args.num_heads, bias=False)
        self.rope = RotaryPositionalEncoding(args.d_rope, max_seq_len=args.max_seq_len)

        # Output Projection
        self.W_o = nn.Linear(args.d_model, args.d_model, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    ## NEW ##: Added position_offset for cached inference
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, position_offset: int = 0):
        B, S, D = x.shape
        past_len = past_kv[0].shape[1] if past_kv is not None else 0

        # --- Path A: Content Path ---
        q_c = self.W_q_content(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        c_kv_new = self.W_dkv_content(x)
        c_kv = torch.cat([past_kv[0], c_kv_new], dim=1) if past_kv is not None else c_kv_new
        k_c = self.W_uk_content(c_kv).view(B, past_len + S, self.num_heads, self.d_head).transpose(1, 2)
        v_c = self.W_uv_content(c_kv).view(B, past_len + S, self.num_heads, self.d_head).transpose(1, 2)
        
        # --- Path B: Position Path ---
        k_r_unrotated = self.W_k_pos(x).view(B, S, self.num_heads, self.d_rope).transpose(1, 2)
        q_r_unrotated = self.W_q_pos(x).view(B, S, self.num_heads, self.d_rope).transpose(1, 2)
        
        k_r_new = self.rope(k_r_unrotated, position_offset=position_offset)
        q_r = self.rope(q_r_unrotated, position_offset=position_offset)

        k_r = torch.cat([past_kv[1], k_r_new], dim=2) if past_kv is not None else k_r_new

        # --- Combining Paths ---
        content_scores = (q_c @ k_c.transpose(-2, -1)) / math.sqrt(self.d_head)
        position_scores = (q_r @ k_r.transpose(-2, -1)) / math.sqrt(self.d_rope)
        
        attn_scores = content_scores + position_scores
        attn_scores = attn_scores + attn_mask
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vector = (attn_weights @ v_c).transpose(1, 2).contiguous().view(B, S, D)
        output = self.W_o(context_vector)
        new_cache = (c_kv, k_r)
        
        return output, new_cache

# --- Mixture-of-Experts (MoE) Modules ---

class ExpertFFN(nn.Module):
    # Same as before
    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

class DeepSeekMoE(nn.Module):
    # Same as before
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_routed = args.moe_n_routed_experts
        self.top_k = args.moe_top_k
        self.routed_experts = nn.ModuleList([ExpertFFN(args.d_model, args.moe_routed_hidden) for _ in range(self.n_routed)])
        self.shared_experts = nn.ModuleList([ExpertFFN(args.d_model, args.moe_routed_hidden) for _ in range(args.moe_n_shared_experts)])
        self.gate = nn.Linear(args.d_model, self.n_routed, bias=False)
        self.register_buffer("bias", torch.zeros(self.n_routed))
        self.bias_lr = 0.01

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)
        shared_out = torch.zeros_like(x)
        for exp in self.shared_experts: shared_out += exp(x)
        router_logits = self.gate(x_flat)
        router_logits_with_bias = router_logits + self.bias.to(router_logits.dtype)
        top_k_logits, top_k_indices = torch.topk(router_logits_with_bias, self.top_k, dim=-1)
        gates = F.softmax(top_k_logits, dim=-1, dtype=torch.float).type_as(x)
        routed_out_flat = torch.zeros_like(x_flat)
        for i in range(self.n_routed):
            mask = (top_k_indices == i); row_idx, which_k = torch.where(mask)
            if row_idx.numel() == 0: continue
            w = gates[row_idx, which_k].unsqueeze(-1)
            exp_in = x_flat[row_idx]
            exp_out = self.routed_experts[i](exp_in)
            routed_out_flat.index_add_(0, row_idx, exp_out * w)
        if self.training:
            with torch.no_grad():
                avg_load = x_flat.size(0) * self.top_k / self.n_routed
                counts = torch.bincount(top_k_indices.flatten(), minlength=self.n_routed).float()
                violation = (avg_load - counts) / (avg_load + 1e-6)
                self.bias.add_(self.bias_lr * torch.tanh(violation))
        routed_out = routed_out_flat.view(B, S, D)
        return shared_out + routed_out

# --- Multi-Token Prediction (MTP) Module ---

class MTPModule(nn.Module):
    # This MTP implementation is simplified for clarity in the main model.py
    # A more complex version might have different projection sizes, etc.
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.projection = nn.Linear(args.d_model * 2, args.d_model, bias=False)
        self.block = TransformerBlock(args)
        
    def forward(self, h_prev, next_token_embeds):
        x = torch.cat([h_prev, next_token_embeds], dim=-1)
        x = self.projection(x)
        
        B, S, D = x.shape
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        attn_mask = torch.zeros(S,S, device=x.device).masked_fill(mask, float('-inf'))
        # We don't use KV cache inside MTP during training.
        h_k, _ = self.block(x, attn_mask=attn_mask.unsqueeze(0).unsqueeze(0))
        return h_k

# --- The Transformer Block ---

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm1 = nn.LayerNorm(args.d_model)
        self.attention = DeepSeekAttention(args)
        self.norm2 = nn.LayerNorm(args.d_model)
        self.feed_forward = DeepSeekMoE(args)

    ## NEW ##: Added position_offset for cached inference
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, position_offset: int = 0):
        h, new_cache = self.attention(self.norm1(x), attn_mask, past_kv, position_offset)
        x = x + h
        x = x + self.feed_forward(self.norm2(x))
        return x, new_cache

# --- The Main Model: MiniDeepSeek ---

class MiniDeepSeek(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm_f = nn.LayerNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.mtp_modules = nn.ModuleList([MTPModule(args) for _ in range(args.n_mtp_modules)])
        
    def causal_mask(self, S: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
        return torch.zeros(S, S, device=device).masked_fill(mask, float('-inf'))

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None, mtp_weight: float = 0.1, past_kv_cache: Optional[list] = None):
        B, S = input_ids.shape
        x = self.embed(input_ids)
        
        # --- Inference Path (with KV Cache) ---
        if targets is None:
            ## NEW ##: Calculate position offset based on past cache
            position_offset = past_kv_cache[0][0].shape[1] if past_kv_cache else 0
            
            mask = self.causal_mask(S + position_offset, x.device)[position_offset:, :]
            mask = mask.unsqueeze(0).unsqueeze(0)

            new_kv_cache = []
            for i, blk in enumerate(self.blocks):
                layer_past_kv = past_kv_cache[i] if past_kv_cache else None
                x, new_cache = blk(x, attn_mask=mask, past_kv=layer_past_kv, position_offset=position_offset)
                new_kv_cache.append(new_cache)
            
            x = self.norm_f(x)
            logits = self.lm_head(x[:, [-1], :]) # Only compute for the last token
            return logits, new_kv_cache

        # --- Training Path (with MTP) ---
        mask = self.causal_mask(S, x.device).unsqueeze(0).unsqueeze(0)
        for blk in self.blocks:
            x, _ = blk(x, attn_mask=mask)
        
        h_main = self.norm_f(x)
        logits_main = self.lm_head(h_main)
        out = {"logits": logits_main}

        main_logits_shift = logits_main[:, :-1, :].contiguous()
        targets_shift = targets[:, 1:].contiguous()
        loss_main = F.cross_entropy(main_logits_shift.view(-1, main_logits_shift.size(-1)), targets_shift.view(-1))
        total_loss = loss_main

        h_prev = h_main
        for k, mtp_block in enumerate(self.mtp_modules, start=1):
            if S <= k + 1: break
            h_prev_shifted = h_prev[:, :-(k+1), :].contiguous()
            next_tokens_embed = self.embed(input_ids[:, k:-1]).contiguous()
            h_k = mtp_block(h_prev_shifted, next_tokens_embed)
            logits_k = self.lm_head(self.norm_f(h_k))
            mtp_targets = targets[:, k+1:].contiguous()
            loss_k = F.cross_entropy(logits_k.reshape(-1, logits_k.size(-1)), mtp_targets.reshape(-1))
            total_loss += mtp_weight * loss_k
            
        out["loss"] = total_loss
        return out

# --- Example Instantiation for testing ---
if __name__ == '__main__':
    args = ModelArgs(d_model=128, n_layers=4, num_heads=4, d_latent=32, d_rope=16, moe_n_routed_experts=4, vocab_size=1000)
    model = MiniDeepSeek(args)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    print("\n--- Testing Training Forward Pass ---")
    dummy_input = torch.randint(0, 1000, (2, 64))
    output = model(dummy_input, targets=dummy_input)
    print("Logits shape:", output['logits'].shape)
    print("Loss value:", output['loss'].item())

    print("\n--- Testing Inference Forward Pass with KV Cache ---")
    model.eval()
    with torch.no_grad():
        prompt = torch.randint(0, 1000, (1, 10))
        logits, kv_cache = model(prompt)
        print("Initial prompt processed. Logits shape:", logits.shape)
        print(f"  - c_kv shape:", kv_cache[0][0].shape)
        
        next_token = torch.argmax(logits, dim=-1)
        
        # ## NEW ##: Pass the old cache to the model
        logits_next, kv_cache_next = model(next_token, past_kv_cache=kv_cache)
        
        print("\nNext token generated. Logits shape:", logits_next.shape)
        print("  - Updated c_kv shape:", kv_cache_next[0][0].shape) # Should be seq_len 11
        assert kv_cache_next[0][0].shape[1] == 11
        print("KV Cache update is correct.")