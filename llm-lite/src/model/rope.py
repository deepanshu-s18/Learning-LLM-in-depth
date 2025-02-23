"""
src/model/rope.py
Notebook 07: Rotary Positional Encoding (RoPE).
Implements complex 2D rotary embedding matrices applied to Query and Key representations,
ensuring relative distance invariance as used in modern models (LLaMA 2/3, DeepSeek).
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

def precompute_rope_frequencies(dim: int, max_seq_len: int, theta_base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes cosine and sine frequency tables for RoPE rotation.
    dim must be even (head_dim).
    theta_i = theta_base ** (-2 * (i - 1) / dim)
    """
    assert dim % 2 == 0, f"Dimension {dim} must be even for RoPE."
    half_dim = dim // 2
    indices = torch.arange(0, half_dim, dtype=torch.float32)
    freqs = 1.0 / (theta_base ** (2.0 * indices / dim))
    
    positions = torch.arange(0, max_seq_len, dtype=torch.float32)
    # Outer product: [max_seq_len, half_dim]
    angles = torch.outer(positions, freqs)
    
    # Repeat along last dim so shape is [max_seq_len, dim]
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """
    Applies rotary positional embeddings to input tensor x.
    x shape: [batch_size, seq_len, num_heads, head_dim] or [batch_size, num_heads, seq_len, head_dim]
    """
    # If shape is [B, n_head, T, head_dim]
    if x.dim() == 4:
        B, n_head, T, head_dim = x.shape
        cos_slice = cos[start_pos : start_pos + T, :head_dim].unsqueeze(0).unsqueeze(0) # [1, 1, T, head_dim]
        sin_slice = sin[start_pos : start_pos + T, :head_dim].unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Expected 4D tensor for RoPE, got {x.dim()}D.")

    # Rotate pairs: [-x1, x0, -x3, x2, ...]
    # Reshape to [..., head_dim // 2, 2]
    x_pairs = x.view(*x.shape[:-1], head_dim // 2, 2)
    x_rotated = torch.stack([-x_pairs[..., 1], x_pairs[..., 0]], dim=-1).flatten(-2)
    
    return x * cos_slice.to(x.device) + x_rotated * sin_slice.to(x.device)


def verify_rope_relative_invariance() -> Dict[str, Any]:
    """
    Mathematically verifies that <RoPE(q, m), RoPE(k, n)> depends strictly
    on relative offset (m - n), remaining identical regardless of absolute position shift.
    """
    dim = 64
    max_len = 100
    cos, sin = precompute_rope_frequencies(dim, max_len)
    
    torch.manual_seed(42)
    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)
    
    # Offset d = 5
    # Pair A: positions (10, 15) -> relative diff = -5
    q_10 = apply_rope(q, cos, sin, start_pos=10)
    k_15 = apply_rope(k, cos, sin, start_pos=15)
    dot_a = torch.sum(q_10 * k_15).item()
    
    # Pair B: positions (50, 55) -> relative diff = -5
    q_50 = apply_rope(q, cos, sin, start_pos=50)
    k_55 = apply_rope(k, cos, sin, start_pos=55)
    dot_b = torch.sum(q_50 * k_55).item()
    
    diff = abs(dot_a - dot_b)
    return {
        "dot_product_pair_A": dot_a,
        "dot_product_pair_B": dot_b,
        "absolute_difference": diff,
        "invariance_verified": diff < 1e-4
    }
