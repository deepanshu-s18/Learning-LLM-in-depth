"""
Minimal GRPO + RLVR building blocks for Chapter 7.

This file collects the small snippets shown in the chapter into one runnable
script. It is not a production trainer; it is a compact reference
implementation for the core mechanics:

1. deterministic verifier rewards
2. group-relative advantages
3. clipped GRPO loss with reference-policy KL
4. one training-step skeleton
"""

from __future__ import annotations

import re
from typing import Iterable

import torch


def extract_final_number(text: str) -> str | None:
    """Return the last integer or decimal number found in a completion."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return matches[-1] if matches else None


def verify_math_answer(completion: str, gold: str) -> float:
    """A toy RLVR reward for answer-only math tasks."""
    predicted = extract_final_number(completion)
    if predicted is None:
        return 0.0
    return 1.0 if predicted == str(gold) else 0.0


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize rewards within each prompt group.

    Args:
        rewards: Tensor of shape [batch_size, group_size].
        eps: Minimum standard deviation for numerical stability.

    Returns:
        Tensor of normalized advantages with the same shape as rewards.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True).clamp_min(eps)
    return (rewards - mean) / std


def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    eps: float = 0.2,
    beta: float = 0.04,
) -> torch.Tensor:
    """Compute the clipped GRPO objective as a minimization loss.

    Shapes:
        logp, old_logp, ref_logp: [batch_size, group_size, tokens]
        advantages: [batch_size, group_size]
    """
    ratio = torch.exp(logp - old_logp)
    clipped = ratio.clamp(1.0 - eps, 1.0 + eps)
