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


class RoPE(nn.Module):
    def __init__(self, d_head, max_seq_len=2048):
        super().__init__()
        self.d_head = d_head
        theta = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("theta", theta)
        positions = torch.arange(max_seq_len)
        freqs = torch.outer(positions, self.theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x, offset=0):
        seq_len = x.shape[2]
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs = self.freqs_cis[offset: offset + seq_len].unsqueeze(0).unsqueeze(0)
        out = torch.view_as_real(x_complex * freqs).flatten(3)
        return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_heads
        self.d_head    = args.d_model // args.num_heads
        self.d_latent  = args.d_latent
