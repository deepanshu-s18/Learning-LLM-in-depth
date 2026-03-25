# **Tutorial Outline: LoRA for LLMs From Scratch**

## **Chapter 1: The Promise - Master LoRA in 30 Minutes**

You've heard of LoRA. It's the key to fine-tuning massive LLMs on a single GPU. You've seen the acronyms: PEFT, low-rank adaptation. But what is it, *really*?

It's not a complex theory. It's a simple, elegant trick.

Instead of training a 1-billion-parameter weight matrix `W`, you freeze it. You then train two tiny matrices, `A` and `B`, that represent the *change* to `W`.

This is LoRA. It's this piece of code:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float = 16.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r

        # Freeze the original linear layer
        self.base = base
