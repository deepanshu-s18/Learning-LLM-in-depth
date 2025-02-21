"""
src/model/attention.py
Notebook 04 & 06: Scaled Dot-Product & Multi-Head Attention with RoPE and KV-Cache Support.
Implements multi-head projection, causal masking, scaling factor 1 / sqrt(d_k),
and dynamic key-value caching for O(1) single-step autoregressive decoding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from src.model.rope import apply_rope

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention layer.
    Supports both training mode (full causal mask) and inference mode with KV-Cache.
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})."
        
        self.n_embd = n_embd
