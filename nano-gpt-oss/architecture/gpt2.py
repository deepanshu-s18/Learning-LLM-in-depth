
import torch
from torch import nn

GPT_CONFIG={'vocab_size': 201088,
 'context_length': 4000,
 'emb_dim': 1260,
 'n_heads': 12,
 'n_layers': 12,
 'drop_rate': 0.1,
 'qkv_bias': False}

class LayerNorm(nn.Module):
    def __init__(self, embd, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embd))
        self.bias = nn.Parameter(torch.zeros(embd))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var= x.var(dim=-1, keepdim=True)
        norm= (x-mean)/torch.sqrt(var+self.eps)
        return norm*self.weight + self.bias
