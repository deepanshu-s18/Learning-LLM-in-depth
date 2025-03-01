"""
src/alignment/reward_model.py
Notebook 12 (Part 1): Bradley-Terry Reward Modeling.
Implements scalar scoring of prompt-completion pairs and trains on pairwise human preferences
using the Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected)).
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from src.model.transformer import TransformerLM
from src.optimization.custom_adamw import CustomAdamW
from data.dataset import SimpleTokenizer

class RewardModel(nn.Module):
    """
    Reward Model wrapping the transformer backbone with a scalar reward head.
    Scores a prompt-completion sequence: r(x, y) in R.
    """
