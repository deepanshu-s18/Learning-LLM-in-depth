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
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_rope = use_rope

        # Fused Q, K, V projection
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask buffer: lower triangular matrix
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Multi-Head Self-Attention.
        
        Args:
            x: Input tensor [B, T, C]
            cos, sin: Precomputed RoPE frequency tables
            start_pos: Position index for RoPE
            kv_cache: Optional tuple of (past_k, past_v) from previous decode steps
        """
        B, T, C = x.size()

        # Calculate Q, K, V projections
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to queries and keys
        if self.use_rope and cos is not None and sin is not None:
            q = apply_rope(q, cos, sin, start_pos=start_pos)
            k = apply_rope(k, cos, sin, start_pos=start_pos)

        # KV-Cache integration (Notebook 06)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv_cache = (k, v)

        # Scaled Dot-Product Attention: Softmax((Q @ K.T) / sqrt(head_dim) + mask) @ V
        total_T = k.size(2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Apply causal mask if processing full sequences (training or prompt prefill)
        if kv_cache is None or T > 1:
            att = att.masked_fill(self.bias[:, :, :T, :total_T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v # [B, n_head, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble heads: [B, T, C]

        return self.resid_dropout(self.c_proj(y)), new_kv_cache
