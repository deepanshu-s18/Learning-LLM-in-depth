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
