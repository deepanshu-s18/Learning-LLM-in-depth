"""
src/tuning/pretrain.py
Notebook 08: LLM Pretraining From Scratch.
Implements Causal Language Modeling (CLM) with next-token prediction, cross-entropy loss,
perplexity calculation, and pretraining training loop.
"""

import math
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
from src.model.transformer import TransformerLM
from src.optimization.custom_adamw import CustomAdamW
from data.dataset import SimpleTokenizer

def prepare_pretrain_batch(texts: List[str], tokenizer: SimpleTokenizer, block_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes text into causal next-token prediction input-target pairs:
    x = tokens[t], y = tokens[t+1]
    """
    token_stream: List[int] = []
    for text in texts:
        token_stream.extend(tokenizer.encode(text))
        token_stream.append(tokenizer.end_token_id)
        
    # Chunk into sequences of length block_size + 1
    inputs = []
    targets = []
    for i in range(0, len(token_stream) - block_size, block_size // 2):
        chunk = token_stream[i : i + block_size + 1]
        if len(chunk) == block_size + 1:
            inputs.append(chunk[:-1])
            targets.append(chunk[1:])
            
    if not inputs:
        # Fallback padding
        inputs = [[tokenizer.pad_token_id] * block_size]
        targets = [[tokenizer.pad_token_id] * block_size]

    x = torch.tensor(inputs, dtype=torch.long, device=device)
    y = torch.tensor(targets, dtype=torch.long, device=device)
    return x, y


def train_pretrain(
    model: TransformerLM,
    texts: List[str],
    tokenizer: SimpleTokenizer,
    epochs: int = 25,
    lr: float = 3e-3,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Executes pretraining on causal text sequences using CustomAdamW.
    """
    model.train()
    optimizer = CustomAdamW(model.parameters(), lr=lr, weight_decay=0.01)
    x_batch, y_batch = prepare_pretrain_batch(texts, tokenizer, model.config.block_size, device)

    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        _, loss, _ = model(x_batch, targets=y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_val = float(loss.item())
        loss_history.append(loss_val)

    final_loss = loss_history[-1]
    perplexity = math.exp(min(final_loss, 20.0))
    
    return {
        "final_pretrain_loss": final_loss,
        "pretrain_perplexity": perplexity,
        "loss_history": loss_history
    }
