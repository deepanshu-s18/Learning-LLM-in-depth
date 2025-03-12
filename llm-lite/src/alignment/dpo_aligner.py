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
    shift_labels = input_ids[:, 1:] # [B, T-1]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather log prob of true tokens
    per_token_log_probs = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    # Mask out prompt tokens so we only sum log-probs over the completion
    completion_log_probs = (per_token_log_probs * label_mask).sum(dim=-1)
    return completion_log_probs


def compute_dpo_loss(
    pi_logps_chosen: torch.Tensor,
    pi_logps_rejected: torch.Tensor,
    ref_logps_chosen: torch.Tensor,
    ref_logps_rejected: torch.Tensor,
    beta: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DPO Loss:
    L = -E[log(sigmoid(beta * (log(pi_w / ref_w) - log(pi_l / ref_l))))]
    """
    pi_logratios = pi_logps_chosen - pi_logps_rejected
    ref_logratios = ref_logps_chosen - ref_logps_rejected

    logits = beta * (pi_logratios - ref_logratios)
