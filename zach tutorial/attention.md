# **Give Me 20 Minutes, I Will Make Attention Click Forever**

## Introduction: The Core Problem & Our Blueprint

The word "**bank**" in "I sat on the river **bank**" is completely different from the "**bank**" in "I withdrew money from the **bank**." For a machine, this is a huge problem. How can the representation of a word change based on its neighbors?

The answer is the **Attention Mechanism**. In the next 20 minutes, you will understand exactly how it works. You will learn:

*   **Word Embeddings:** The static starting point for every word.
*   **Scaled Dot-Product Attention:** The core formula that enables context.
*   **Query, Key, and Value (QKV):** The three roles a word can play.
*   **The Causal Mask:** How to prevent the model from cheating by looking ahead.
*   **Multi-Head Attention:** How to scale the mechanism for powerful models.

This is the entire secret. This single formula and the Python code that implements it are the engine behind models like ChatGPT.

**The Formula:**
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V $$

**The Code:**
```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head, self.n_embd = config.n_head, config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
```
This might look intimidating, but we will build it from the ground up until every line is obvious. But before we can build this engine, we must first understand the fuel it runs on: word vectors.

---

