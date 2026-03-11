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

### **Chapter 1: The Starting Point & Its Fatal Flaw (Word Embeddings)**

A neural network cannot understand the word "cat". It can only understand lists of numbers, called **vectors**. Our first job is to convert every word in our vocabulary into a unique vector.

The mechanism for this is a simple lookup table called an **Embedding Layer**.

**The Mechanism: A Learnable Dictionary**
Imagine a giant spreadsheet with one row for every word in the vocabulary. Each row contains the vector for that word. The `nn.Embedding` layer is exactly this.

```python
import torch
import torch.nn as nn

# A tiny config for our example
vocab_size = 10    # Our dictionary has 10 words
n_embd = 4         # Each word will be represented by a vector of size 4

# The layer is our coordinate book
token_embedding_table = nn.Embedding(vocab_size, n_embd)

# Let's look up the vector for the word with ID=3
input_id = torch.tensor([3])
vector = token_embedding_table(input_id)

print(f"The vector for word ID {input_id.item()} is:\n{vector}")
```
**Output:**
```
The vector for word ID 3 is:
tensor([[-1.5323, -0.2343,  0.5132, -1.0833]], grad_fn=<EmbeddingBackward0>)
```
Initially, these vectors are random. During training, the model learns the optimal vector for each word.

**The Fatal Flaw: No Context**

This simple lookup has one massive problem: it is **static**. The vector for a word is the same regardless of the words around it.

Let's return to our "bank" example.
*   Sentence 1: "I sat on the river **bank**."
*   Sentence 2: "I withdrew money from the **bank**."

Let's assume the word "bank" has ID `7` in our vocabulary. When we look up its vector, the process is identical for both sentences.

| Context | Word | Lookup Process | Resulting Vector |
| :--- | :--- | :--- | :--- |
| "river..." | bank | `embedding_table[7]` | `[0.1, 0.8, -0.4, ...]` |
| "money..." | bank | `embedding_table[7]` | `[0.1, 0.8, -0.4, ...]` **(Identical!)** |

This is the core limitation. Our initial vectors are context-free. They represent a word's general meaning but are blind to the specific meaning in a sentence.

