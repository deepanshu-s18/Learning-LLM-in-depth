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
