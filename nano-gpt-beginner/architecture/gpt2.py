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

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out    = d_out
        self.num_heads= num_heads
        self.head_dim = d_out // num_heads

        self.W_query  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key    = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout  = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, n, d_in = x.shape
        q = self.W_query(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(2, 3)
        scores.masked_fill_(self.mask.bool()[:n, :n], -torch.inf)
        weights = self.dropout(torch.softmax(scores / k.shape[-1]**0.5, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(b, n, self.d_out)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn    = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"],
                                          cfg["context_length"], cfg["drop_rate"],
                                          cfg["n_heads"], cfg["qkv_bias"])
        self.ln1     = LayerNorm(cfg["emb_dim"])
        self.ln2     = LayerNorm(cfg["emb_dim"])
        self.ff      = FeedForward(cfg)
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb   = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb   = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb  = nn.Dropout(cfg["drop_rate"])
        self.blocks    = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm= LayerNorm(cfg["emb_dim"])
        self.out_head  = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, idx):
        b, s = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(s, device=idx.device))
        x = self.drop_emb(x)
        x = self.blocks(x)
        return self.out_head(self.final_norm(x))
