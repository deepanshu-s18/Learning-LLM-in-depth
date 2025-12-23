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
        inputs = tokenizer(partial, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = reward_model(**inputs)
            if hasattr(outputs, "logits"):
                if outputs.logits.numel() == 1:
                    score = outputs.logits[0].item()
                else:
                    score = outputs.logits[0][0].item()
            else:
                score = float(outputs[0])
        cumulative_score += score

    return cumulative_score / len(steps) if steps else 0.0


def beam_search_with_prm(
    prompt: str,
    reasoning_model,
    reasoning_tokenizer,
    reward_model,
    reward_tokenizer,
    is_seq2seq: bool = False,
    N: int = 4,
    M: int = 2,
    max_steps: int = 3,
    max_new_tokens: int = 64,
    device=None,
):
    """
    Performs PRM-guided Beam Search over multi-step reasoning generation.
    Supports both Seq2Seq (Flan-T5) and Causal LMs (Zephyr, TinyLlama).
    """
    assert N % M == 0, f"N ({N}) must be divisible by M ({M})"
    if device is None:
        device = next(reasoning_model.parameters()).device

    # Format prompt
    if is_seq2seq:
        formatted_prompt = (
            f"Question: {prompt}\n"
            f"Answer: Let's think step by step."
        )
    elif hasattr(reasoning_tokenizer, "chat_template") and reasoning_tokenizer.chat_template:
        messages = [
            {"role": "system", "content": "You are a helpful reasoning assistant that solves problems step-by-step."},
