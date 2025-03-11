"""
src/alignment/dpo_aligner.py
Notebook 13: Direct Preference Optimization (DPO).
Directly optimizes policy probabilities on pairwise human preferences without training
a separate reward model or executing an unstable reinforcement learning loop.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from src.model.transformer import TransformerLM
from src.optimization.custom_adamw import CustomAdamW
from data.dataset import SimpleTokenizer

def get_batch_log_probs(model: TransformerLM, input_ids: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes sum of log-probabilities of tokens where label_mask is 1.
    input_ids: [B, T]
    label_mask: [B, T-1]
    """
    logits, _, _ = model(input_ids) # [B, T, V]
    shift_logits = logits[:, :-1, :] # [B, T-1, V]
