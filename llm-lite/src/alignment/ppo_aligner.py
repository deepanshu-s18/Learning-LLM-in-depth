"""
src/alignment/ppo_aligner.py
Notebook 12 (Part 2): Proximal Policy Optimization (PPO) with Actor-Critic and KL Penalty.
Implements policy optimization anchored by a frozen reference model to maximize reward
while preventing catastrophic policy drift.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from src.model.transformer import TransformerLM
from src.alignment.reward_model import RewardModel
from src.optimization.custom_adamw import CustomAdamW
from data.dataset import SimpleTokenizer

class ValueCritic(nn.Module):
    """
    Critic network V_psi estimating expected return from sequence states.
    """
    def __init__(self, base_lm: TransformerLM):
