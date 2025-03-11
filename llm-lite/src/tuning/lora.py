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
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False

        # Freeze the base weight & bias
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))

        # Initialization: Kaiming uniform for A, zero for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)
        # Base forward + low-rank delta
        base_out = self.base_layer(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

    def merge_weights(self) -> None:
        """Fuses LoRA delta directly into base weights for zero-overhead deployment."""
        if not self.merged:
            with torch.no_grad():
                delta_W = (self.lora_B @ self.lora_A) * self.scaling
                self.base_layer.weight.data.add_(delta_W)
            self.merged = True


def inject_lora(model: nn.Module, r: int = 4, alpha: float = 8.0) -> int:
    """
    Recursively replaces attention projection layers with LoRALinear wrappers.
    Returns the total number of trainable LoRA parameters.
    """
    trainable_params = 0
    for name, module in model.named_modules():
        if hasattr(module, "c_attn") and isinstance(module.c_attn, nn.Linear):
            module.c_attn = LoRALinear(module.c_attn, r=r, alpha=alpha)
            trainable_params += (module.c_attn.lora_A.numel() + module.c_attn.lora_B.numel())
        if hasattr(module, "c_proj") and isinstance(module.c_proj, nn.Linear):
            module.c_proj = LoRALinear(module.c_proj, r=r, alpha=alpha)
            trainable_params += (module.c_proj.lora_A.numel() + module.c_proj.lora_B.numel())
    return trainable_params


def get_parameter_summary(model: nn.Module) -> Dict[str, Any]:
    """Computes total vs trainable parameter counts and parameter reduction ratio."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reduction_pct = 100.0 * (1.0 - (trainable_params / max(total_params, 1)))
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_percentage": reduction_pct
    }
