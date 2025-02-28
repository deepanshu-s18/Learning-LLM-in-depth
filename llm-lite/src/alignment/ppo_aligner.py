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
        super().__init__()
        self.backbone = copy.deepcopy(base_lm)
        self.value_head = nn.Linear(base_lm.config.n_embd, 1, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.size()
        x = self.backbone.drop(self.backbone.wte(input_ids))
        for block in self.backbone.blocks:
            x, _ = block(x, cos=self.backbone.rope_cos, sin=self.backbone.rope_sin)
        x = self.backbone.ln_f(x)
        return self.value_head(x[:, -1, :]).squeeze(-1) # [B]


def train_ppo_alignment(
    actor_policy: TransformerLM,
    reward_model: RewardModel,
    prompts: List[str],
    tokenizer: SimpleTokenizer,
    epochs: int = 10,
    lr: float = 5e-4,
    kl_coef: float = 0.1,
    clip_eps: float = 0.2,
