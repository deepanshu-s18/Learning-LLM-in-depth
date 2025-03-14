"""
src/tuning/sft.py
Notebook 09: Supervised Fine-Tuning (SFT) with Prompt Loss Masking.
Formats conversational interactions with special tokens (<|user|>, <|assistant|>, <|end|>)
and applies target masking (label = -100) so gradients only flow through assistant tokens.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any
from src.model.transformer import TransformerLM
from src.optimization.custom_adamw import CustomAdamW
from data.dataset import SimpleTokenizer

def prepare_sft_batch(
    dataset: List[Dict[str, str]],
    tokenizer: SimpleTokenizer,
    block_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates input tokens and masked target tokens.
    Target tokens corresponding to the user prompt are masked with -100 (ignored by CrossEntropyLoss).
    """
    input_list = []
    target_list = []

    for item in dataset:
        prompt_text = f"<|user|>{item['prompt']}<|assistant|>"
        response_text = f"{item['response']}<|end|>"

        prompt_tokens = tokenizer.encode(prompt_text)
        response_tokens = tokenizer.encode(response_text)

        full_sequence = prompt_tokens + response_tokens
        # Labels: mask prompt tokens with -100
        labels = ([-100] * len(prompt_tokens)) + response_tokens

        # Truncate or pad to block_size + 1
        if len(full_sequence) > block_size + 1:
            full_sequence = full_sequence[: block_size + 1]
            labels = labels[: block_size + 1]
