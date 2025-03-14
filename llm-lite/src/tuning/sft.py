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
        else:
            pad_len = (block_size + 1) - len(full_sequence)
            full_sequence = full_sequence + ([tokenizer.pad_token_id] * pad_len)
            labels = labels + ([-100] * pad_len)

        # Causal shift: input is seq[:-1], target is seq[1:]
        input_list.append(full_sequence[:-1])
        target_list.append(labels[1:])

    x = torch.tensor(input_list, dtype=torch.long, device=device)
    y = torch.tensor(target_list, dtype=torch.long, device=device)
    return x, y


def train_sft(
    model: TransformerLM,
    dataset: List[Dict[str, str]],
    tokenizer: SimpleTokenizer,
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Executes Supervised Fine-Tuning with prompt loss masking.
    """
    model.train()
    optimizer = CustomAdamW(model.parameters(), lr=lr, weight_decay=0.0)
    x_batch, y_batch = prepare_sft_batch(dataset, tokenizer, model.config.block_size, device)

    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        _, loss, _ = model(x_batch, targets=y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_history.append(float(loss.item()))

    return {
        "final_sft_loss": loss_history[-1],
        "loss_history": loss_history
    }
