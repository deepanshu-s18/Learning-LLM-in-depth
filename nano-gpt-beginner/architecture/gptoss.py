import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 201088
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    intermediate_size: int = 2048
    num_experts: int = 4
    experts_per_token: int = 1
    max_position_embeddings: int = 4096
    dropout: float = 0.1


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * x / rms


class GQA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads   = cfg.num_attention_heads
        self.n_kv      = cfg.num_key_value_heads
        self.head_dim  = cfg.hidden_size // self.n_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj  = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj  = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj  = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj  = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
