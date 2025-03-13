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
