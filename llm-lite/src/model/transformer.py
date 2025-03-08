"""
src/model/transformer.py
Notebook 05: Transformer From Scratch (Decoder-Only Architecture).
Implements modern Pre-LN transformer architecture featuring RMSNorm, GELU MLP blocks,
Multi-Head Attention with RoPE, and Tied Language Modeling Head.
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any

from src.model.attention import MultiHeadAttention
from src.model.rope import precompute_rope_frequencies

@dataclass
class TransformerConfig:
    vocab_size: int = 128
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    use_rope: bool = True


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    Computationally faster than LayerNorm by omitting mean-centering.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2) + eps)
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class MLP(nn.Module):
    """Feed-forward network with 4x expansion and GELU activation."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Decoder Block with residual connections."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
