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
