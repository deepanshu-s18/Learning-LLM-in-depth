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
