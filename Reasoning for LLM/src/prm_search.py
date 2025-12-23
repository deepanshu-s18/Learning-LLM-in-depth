"""
Process Reward Model (PRM) Guided Beam Search for Step-by-Step LLM Reasoning.

Implements inference-time compute scaling via search over reasoning steps,
scoring partial reasoning traces using a PRM / sequence classification reward model,
and visualizing the resulting reasoning tree graph.
"""

import os
import textwrap
import torch
import networkx as nx
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def get_default_device():
    """Detect available hardware accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stepwise_prm_score(prompt: str, trace: str, reward_model, tokenizer, device=None) -> float:
    """
    Simulate stepwise reward using an outcome/process reward model.
    Evaluates intermediate prefixes of reasoning steps and averages cumulative scores.
    """
    if device is None:
        device = next(reward_model.parameters()).device if hasattr(reward_model, "parameters") else "cpu"

    steps = [s.strip() for s in trace.split(". ") if s.strip()]
    if not steps:
        steps = [trace.strip()] if trace.strip() else [""]

    cumulative_score = 0.0
    for i in range(1, len(steps) + 1):
        partial = prompt + "\n" + ". ".join(steps[:i])
