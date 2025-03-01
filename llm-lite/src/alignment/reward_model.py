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
    def __init__(self, base_lm: TransformerLM):
        super().__init__()
        # Clone backbone
        self.backbone = copy.deepcopy(base_lm)
        hidden_dim = base_lm.config.n_embd
        # Scalar reward head replaces LM vocabulary projection
        self.reward_head = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns scalar reward for the entire sequence (extracted from the final token position).
        input_ids: [B, T]
        """
        B, T = input_ids.size()
        x = self.backbone.drop(self.backbone.wte(input_ids))
        for block in self.backbone.blocks:
            x, _ = block(x, cos=self.backbone.rope_cos, sin=self.backbone.rope_sin)
        x = self.backbone.ln_f(x)
        # Take the hidden representation of the last token
        last_hidden = x[:, -1, :] # [B, hidden_dim]
        rewards = self.reward_head(last_hidden).squeeze(-1) # [B]
        return rewards


def compute_bradley_terry_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry Pairwise Preference Loss:
    L = -E[log(sigmoid(r_chosen - r_rejected))]
    """
    # Numerically stable formulation of -log(sigmoid(diff)) = log(1 + exp(-diff))
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def train_reward_model(
    reward_model: RewardModel,
    preference_data: List[Dict[str, str]],
    tokenizer: SimpleTokenizer,
    epochs: int = 15,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Trains the reward model to rank chosen completions higher than rejected completions.
    """
    reward_model.train()
    optimizer = CustomAdamW(reward_model.parameters(), lr=lr, weight_decay=0.01)
    block_size = reward_model.backbone.config.block_size

    # Prepare batches
    chosen_ids = []
    rejected_ids = []
    for item in preference_data:
        c_text = f"<|user|>{item['prompt']}<|assistant|>{item['chosen']}<|end|>"
        r_text = f"<|user|>{item['prompt']}<|assistant|>{item['rejected']}<|end|>"
        
        c_enc = tokenizer.encode(c_text)[:block_size]
        r_enc = tokenizer.encode(r_text)[:block_size]
        
        # Pad
        c_enc += [tokenizer.pad_token_id] * (block_size - len(c_enc))
        r_enc += [tokenizer.pad_token_id] * (block_size - len(r_enc))
        
        chosen_ids.append(c_enc)
        rejected_ids.append(r_enc)

    c_tensor = torch.tensor(chosen_ids, dtype=torch.long, device=device)
    r_tensor = torch.tensor(rejected_ids, dtype=torch.long, device=device)

    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        r_c = reward_model(c_tensor)
        r_r = reward_model(r_tensor)
        loss = compute_bradley_terry_loss(r_c, r_r)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_history.append(float(loss.item()))

    return {
        "final_rm_loss": loss_history[-1],
        "chosen_rewards_mean": float(r_c.mean().item()),
        "rejected_rewards_mean": float(r_r.mean().item()),
        "loss_history": loss_history
    }
