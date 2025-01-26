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
