import torch
from torch import nn

GPT_CONFIG = {
    'vocab_size': 201088,
    'context_length': 2048,
    'emb_dim': 1024,
    'n_heads': 12,
    'n_layers': 12,
    'drop_rate': 0.1,
    'qkv_bias': False
}

class LayerNorm(nn.Module):
    def __init__(self, embd, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embd))
        self.bias   = nn.Parameter(torch.zeros(embd))
        self.eps    = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return norm * self.weight + self.bias


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4),
            GELU(),
            nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"])
        )

