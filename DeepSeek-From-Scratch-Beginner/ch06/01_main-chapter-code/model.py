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
        self.d_rope    = args.d_rope

        self.wq_c = nn.Linear(args.d_model, args.d_model, bias=False)
        self.wdkv = nn.Linear(args.d_model, args.d_latent, bias=False)
        self.wuk  = nn.Linear(args.d_latent, args.d_model, bias=False)
        self.wuv  = nn.Linear(args.d_latent, args.d_model, bias=False)
        self.wkr  = nn.Linear(args.d_model, args.d_rope * args.num_heads, bias=False)
        self.wqr  = nn.Linear(args.d_model, args.d_rope * args.num_heads, bias=False)
        self.rope  = RoPE(args.d_rope, max_seq_len=args.max_seq_len)
        self.wo    = nn.Linear(args.d_model, args.d_model, bias=False)
        self.drop  = nn.Dropout(args.dropout)

    def forward(self, x, mask, past_kv=None, offset=0):
        B, S, D = x.shape
        past_len = past_kv[0].shape[1] if past_kv else 0

        qc = self.wq_c(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        ckv_new = self.wdkv(x)
        ckv = torch.cat([past_kv[0], ckv_new], dim=1) if past_kv else ckv_new
        kc = self.wuk(ckv).view(B, past_len + S, self.num_heads, self.d_head).transpose(1, 2)
        vc = self.wuv(ckv).view(B, past_len + S, self.num_heads, self.d_head).transpose(1, 2)

        kr_new = self.rope(self.wkr(x).view(B, S, self.num_heads, self.d_rope).transpose(1, 2), offset=offset)
        qr     = self.rope(self.wqr(x).view(B, S, self.num_heads, self.d_rope).transpose(1, 2), offset=offset)
        kr     = torch.cat([past_kv[1], kr_new], dim=2) if past_kv else kr_new

        scores = (qc @ kc.transpose(-2, -1)) / math.sqrt(self.d_head) +                  (qr @ kr.transpose(-2, -1)) / math.sqrt(self.d_rope)
        scores = scores + mask
        attn   = self.drop(F.softmax(scores, dim=-1))
        out    = self.wo((attn @ vc).transpose(1, 2).contiguous().view(B, S, D))
        return out, (ckv, kr)


class Expert(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.0):
        super().__init__()
        self.fc1  = nn.Linear(d_model, hidden, bias=False)
        self.fc2  = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_routed = args.moe_n_routed_experts
        self.top_k    = args.moe_top_k
        self.routed   = nn.ModuleList([Expert(args.d_model, args.moe_routed_hidden) for _ in range(self.n_routed)])
        self.shared   = nn.ModuleList([Expert(args.d_model, args.moe_routed_hidden) for _ in range(args.moe_n_shared_experts)])
        self.gate     = nn.Linear(args.d_model, self.n_routed, bias=False)
        self.register_buffer("bias", torch.zeros(self.n_routed))
        self.bias_lr  = 0.01

    def forward(self, x):
        B, S, D = x.shape
        xf = x.reshape(-1, D)
        shared_out = sum(e(x) for e in self.shared)
        logits  = self.gate(xf) + self.bias.to(xf.dtype)
        topv, topi = torch.topk(logits, self.top_k, dim=-1)
        gates   = F.softmax(topv, dim=-1, dtype=torch.float).type_as(x)
        out     = torch.zeros_like(xf)
        for i in range(self.n_routed):
            mask = (topi == i)
            rows, ks = torch.where(mask)
            if rows.numel() == 0:
                continue
            out.index_add_(0, rows, self.routed[i](xf[rows]) * gates[rows, ks].unsqueeze(-1))
        if self.training:
            with torch.no_grad():
                avg = xf.size(0) * self.top_k / self.n_routed
                cnt = torch.bincount(topi.flatten(), minlength=self.n_routed).float()
                self.bias.add_(self.bias_lr * torch.tanh((avg - cnt) / (avg + 1e-6)))
        return shared_out + out.view(B, S, D)


class MTPBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.proj  = nn.Linear(args.d_model * 2, args.d_model, bias=False)
        self.block = Block(args)

    def forward(self, h_prev, next_embeds):
        x = self.proj(torch.cat([h_prev, next_embeds], dim=-1))
        S = x.shape[1]
        mask = torch.zeros(S, S, device=x.device).masked_fill(
            torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1), float("-inf")
        )
        h, _ = self.block(x, mask.unsqueeze(0).unsqueeze(0))
        return h


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm1 = nn.LayerNorm(args.d_model)
        self.attn  = Attention(args)
        self.norm2 = nn.LayerNorm(args.d_model)
        self.ff    = MoE(args)

    def forward(self, x, mask, past_kv=None, offset=0):
        h, cache = self.attn(self.norm1(x), mask, past_kv, offset)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x, cache


class MiniDeepSeek(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args   = args
        self.embed  = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.norm   = nn.LayerNorm(args.d_model)
        self.head   = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.mtp    = nn.ModuleList([MTPBlock(args) for _ in range(args.n_mtp_modules)])

    def _causal_mask(self, S, device):
        m = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
        return torch.zeros(S, S, device=device).masked_fill(m, float("-inf"))

    def forward(self, ids, targets=None, mtp_w=0.1, past_kv=None):
        B, S = ids.shape
        x = self.embed(ids)
        if targets is None:
            offset = past_kv[0][0].shape[1] if past_kv else 0
            mask   = self._causal_mask(S + offset, x.device)[offset:].unsqueeze(0).unsqueeze(0)
            new_kv = []
            for i, blk in enumerate(self.blocks):
                x, c = blk(x, mask, past_kv[i] if past_kv else None, offset)
                new_kv.append(c)
            return self.head(self.norm(x[:, [-1], :])), new_kv

        mask = self._causal_mask(S, x.device).unsqueeze(0).unsqueeze(0)
        for blk in self.blocks:
            x, _ = blk(x, mask)
        h      = self.norm(x)
        logits = self.head(h)
        loss   = F.cross_entropy(logits[:, :-1].contiguous().view(-1, self.args.vocab_size),
                                 targets[:, 1:].contiguous().view(-1))
        for k, mtp in enumerate(self.mtp, 1):
            if S <= k + 1:
                break
            hk   = mtp(h[:, :-(k+1)], self.embed(ids[:, k:-1]))
            lk   = self.head(self.norm(hk))
            loss += mtp_w * F.cross_entropy(lk.reshape(-1, self.args.vocab_size),
                                             targets[:, k+1:].reshape(-1))
        return {"logits": logits, "loss": loss}
