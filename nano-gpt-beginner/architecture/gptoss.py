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
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv,    self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv,    self.head_dim).transpose(1, 2)

        # repeat kv heads to match q heads
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), diagonal=1)
        attn.masked_fill_(mask, float("-inf"))
        attn = self.dropout(torch.softmax(attn, dim=-1))
        out  = (attn @ v).transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out)


class Expert(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.k       = cfg.experts_per_token
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.num_experts)])
        self.gate    = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)

    def forward(self, x):
        b, s, d = x.shape
        xf  = x.view(-1, d)
        g   = self.gate(xf)
        w, idx = torch.topk(g, self.k, dim=-1)
        w   = torch.softmax(w, dim=-1)
        out = torch.zeros_like(xf)
        for i, exp in enumerate(self.experts):
            mask = (idx == i).any(dim=-1)
            if mask.any():
                which_k = (idx[mask] == i).float()
                gate_w  = (w[mask] * which_k).sum(-1, keepdim=True)
                out[mask] += gate_w * exp(xf[mask])
        return out.view(b, s, d)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1   = RMSNorm(cfg.hidden_size)
        self.attn    = GQA(cfg)
        self.norm2   = RMSNorm(cfg.hidden_size)
        self.moe     = MoE(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.moe(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig, device="cpu"):
        super().__init__()
        self.tok_emb  = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb  = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.blocks   = nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm     = RMSNorm(cfg.hidden_size)
        self.lm_head  = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.to(device)

    def forward(self, idx):
        b, s = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(s, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm(x))
