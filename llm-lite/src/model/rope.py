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
