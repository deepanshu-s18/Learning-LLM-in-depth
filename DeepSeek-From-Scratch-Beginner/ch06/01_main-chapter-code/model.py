import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelArgs:
    d_model: int = 512
    n_layers: int = 6
    vocab_size: int = 50257
    num_heads: int = 4
    d_latent: int = 128
    d_rope: int = 32
    moe_n_routed_experts: int = 8
    moe_n_shared_experts: int = 1
    moe_top_k: int = 2
    moe_routed_hidden: int = 256
    n_mtp_modules: int = 1
    dropout: float = 0.05
    max_seq_len: int = 2048


