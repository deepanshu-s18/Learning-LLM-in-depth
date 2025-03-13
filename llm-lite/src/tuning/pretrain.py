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
