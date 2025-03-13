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
