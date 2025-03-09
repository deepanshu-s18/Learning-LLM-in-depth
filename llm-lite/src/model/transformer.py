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
        self.attn = MultiHeadAttention(
            n_embd=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size,
            dropout=config.dropout,
            use_rope=config.use_rope
        )
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LN Self-Attention
        norm_x = self.ln_1(x)
        attn_out, new_kv_cache = self.attn(norm_x, cos=cos, sin=sin, start_pos=start_pos, kv_cache=kv_cache)
        x = x + attn_out
        
        # Pre-LN MLP
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache


class TransformerLM(nn.Module):
    """
    Complete Decoder-Only Generative Language Model.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        
        # LM Head (projects hidden state back to vocabulary logits)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: share weights between token embeddings and output projection
        self.lm_head.weight = self.wte.weight

        # Precompute RoPE frequency tables
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope_frequencies(head_dim, config.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        idx: [B, T]
        targets: [B, T] (optional, computes cross-entropy loss)
        """
        B, T = idx.size()
        x = self.drop(self.wte(idx))
        
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(
                x,
                cos=self.rope_cos,
                sin=self.rope_sin,
                start_pos=start_pos,
                kv_cache=layer_cache
            )
            new_kv_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x) # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # Shift targets for causal language modeling
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        use_kv_cache: bool = True,
        eos_token_id: Optional[int] = 3
    ) -> List[int]:
        """
        Autoregressive text generation.
        Supports both fast KV-Cache decoding and naive non-cached decoding.
        """
        self.eval()
        device = next(self.parameters()).device
        generated = list(prompt_ids)
        
        if use_kv_cache:
            # Prefill phase with prompt
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            logits, _, kv_caches = self.forward(idx, start_pos=0)
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            # Incremental single-token decode phase
            curr_pos = len(prompt_ids)
            for _ in range(max_new_tokens - 1):
                if eos_token_id is not None and next_token == eos_token_id:
                    break
                idx = torch.tensor([[next_token]], dtype=torch.long, device=device)
                logits, _, kv_caches = self.forward(idx, kv_caches=kv_caches, start_pos=curr_pos)
                next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)
                curr_pos += 1
        else:
            # Naive generation: recompute entire context at every step (O(T^2))
            for _ in range(max_new_tokens):
                context = generated[-self.config.block_size:]
                idx = torch.tensor([context], dtype=torch.long, device=device)
                logits, _, _ = self.forward(idx, start_pos=0)
                next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)
                if eos_token_id is not None and next_token == eos_token_id:
                    break

        return generated
