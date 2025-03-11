"""
src/tuning/lora.py
Notebook 10: LoRA: Low-Rank Adaptation for LLMs From Scratch.
Implements parameter-efficient fine-tuning (PEFT) via low-rank decomposition W + (alpha / r) * B @ A,
base weight freezing, parameter accounting, and zero-overhead weight merging.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

class LoRALinear(nn.Module):
    """
    LoRA wrapper around an nn.Linear layer.
    W_new = W_0 + (alpha / r) * B @ A
    Base weight W_0 is frozen; only A (r x d_in) and B (d_out x r) are trained.
    """
    def __init__(self, base_layer: nn.Linear, r: int = 4, alpha: float = 8.0):
        super().__init__()
