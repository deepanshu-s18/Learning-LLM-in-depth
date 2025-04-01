"""
GRPO + RLVR minimal implementation for chapter 7.
"""

import re
import torch


def get_last_number(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def reward_fn(completion, gold):
    pred = get_last_number(completion)
    if pred is None:
        return 0.0
    return 1.0 if pred == str(gold) else 0.0


def group_advantages(rewards, eps=1e-6):
    mean = rewards.mean(dim=1, keepdim=True)
    std  = rewards.std(dim=1, keepdim=True).clamp_min(eps)
    return (rewards - mean) / std


def grpo_loss(logp, old_logp, ref_logp, advantages, eps=0.1, beta=0.04):
    ratio     = torch.exp(logp - old_logp)
    clipped   = ratio.clamp(1 - eps, 1 + eps)
    adv       = advantages.unsqueeze(-1)
    surrogate = torch.minimum(ratio * adv, clipped * adv)
    kl        = logp - ref_logp
    return -(surrogate - beta * kl).mean()


def demo():
    rewards    = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]])
    advantages = group_advantages(rewards)
    logp       = torch.randn(2, 4, 6) * 0.05
    loss       = grpo_loss(logp, torch.zeros_like(logp), torch.zeros_like(logp), advantages)
    print(f"demo loss: {loss.item():.4f}")


if __name__ == "__main__":
    demo()
