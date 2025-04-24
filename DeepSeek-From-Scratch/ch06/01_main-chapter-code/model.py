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
