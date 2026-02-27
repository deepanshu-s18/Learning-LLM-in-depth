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
        self.base.weight.requires_grad_(False)

        # Create the trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.empty(base.out_features, r))

        # Initialize the weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B) # Start with no change

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original path (frozen) + LoRA path (trainable)
        return self.base(x) + (F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling)

```

**My promise:** You will understand every line of this code, the math behind it, and why it's so effective, in the next 30 minutes. Let's begin.

## **Chapter 2: The Foundation - The `nn.Linear` Layer**

Before we can modify an LLM, we must understand its most fundamental part: the `nn.Linear` layer. It's the simple workhorse that performs the vast majority of computations in a Transformer.

Its only job is to perform this equation: `output = input @ W.T + b`

#### A Minimal, Reproducible Example

Let's see this in action. We'll create a tiny linear layer that takes a vector of size 3 and outputs a vector of size 2. To make this perfectly clear, we will set the weights and bias manually.
