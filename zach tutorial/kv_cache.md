# **Give me 30 min, I will make KV cache click forever**

## **Introduction: The Bottleneck in Our Transformer**

In our last session, we built a complete, working GPT model from scratch. We demystified every component, from embeddings to the final prediction head. The result was this minimal, understandable implementation:

```python
class GPT2(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
```

This code works. It trains, and it generates text. But it has a secret, critical flaw: the generation process is painfully inefficient. Generating text one token at a time is slow, and the reason lies in a massive amount of redundant computation hidden within our self-attention mechanism.

You should be familiar with the core attention formula. If not, I highly recommend watching our previous video on the Transformer architecture.
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V $$
The bottleneck is this: every time we generate a single new token, we re-calculate the **Key (K)** and **Value (V)** vectors for *every single token that came before it*. This is incredibly wasteful.

#### Our Promise and The Solution

In the next 30 minutes, this inefficiency will not only make perfect sense, but you will have fixed it. We are going to implement the **KV Cache**, a simple and elegant optimization that is fundamental to every modern large language model.

The idea is straightforward: instead of re-computing old Key and Value vectors, we will cache them. This one change will dramatically accelerate our model's generation speed by transforming the computational complexity from quadratic to linear.

Here is exactly how we will do it, presented as the code changes we are going to make.

**This is the change we will make to `CausalSelfAttention`:**
```diff
--- a/gpt2_min.py
+++ b/gpt2_min_kv_cache.py
@@ -23,17 +23,26 @@
         self.c_proj = nn.Linear(config.n_embd, config.n_embd)
         self.resid_drop = nn.Dropout(config.dropout)
