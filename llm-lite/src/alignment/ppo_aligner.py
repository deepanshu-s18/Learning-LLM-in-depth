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
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    PPO RLHF Loop:
    1. Frozen Reference Policy pi_ref ensures stability.
    2. Reward Model scores generated responses.
    3. KL divergence penalty keeps actor close to pi_ref.
    4. Clipped surrogate loss updates actor weights.
    """
    actor_policy.train()
    reward_model.eval()
    
    # Frozen reference policy pi_ref
    ref_policy = copy.deepcopy(actor_policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    critic = ValueCritic(actor_policy).to(device)
    critic.train()

    optimizer = CustomAdamW(
        list(actor_policy.parameters()) + list(critic.parameters()),
        lr=lr,
        weight_decay=0.0
    )

    block_size = actor_policy.config.block_size
    loss_history = []
    rewards_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Batch preparation from prompts
        batch_ids = []
        for p_text in prompts:
            full_text = f"<|user|>{p_text}<|assistant|>"
            enc = tokenizer.encode(full_text)[:block_size]
            enc += [tokenizer.pad_token_id] * (block_size - len(enc))
            batch_ids.append(enc)
        
        input_tensor = torch.tensor(batch_ids, dtype=torch.long, device=device)

        # 1. Forward pass on Actor and Frozen Reference
        logits_actor, _, _ = actor_policy(input_tensor)
        with torch.no_grad():
            logits_ref, _, _ = ref_policy(input_tensor)
            raw_rewards = reward_model(input_tensor)

        # 2. Token-level Log Probabilities
        log_probs_actor = F.log_softmax(logits_actor, dim=-1)
        log_probs_ref = F.log_softmax(logits_ref, dim=-1)

        # 3. Approximate KL Divergence: KL(pi_theta || pi_ref) = sum(pi * (log pi - log ref))
        probs_actor = F.softmax(logits_actor, dim=-1)
        kl_div = torch.sum(probs_actor * (log_probs_actor - log_probs_ref), dim=-1).mean(-1) # [B]

        # 4. Penalized Reward: r_penalized = r - beta * KL
        penalized_reward = raw_rewards - (kl_coef * kl_div)

        # 5. Critic Value & Advantage: A = r_penalized - V(s)
        values = critic(input_tensor)
        advantages = (penalized_reward - values).detach()

        # 6. PPO Clipped Surrogate Loss
        # Importance ratio r_t(theta) approx exp(log_prob_actor - log_prob_old)
        ratio = torch.exp((log_probs_actor - log_probs_ref).mean(dim=[1, 2]))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (MSE)
        value_loss = F.mse_loss(values, penalized_reward)

        total_loss = policy_loss + 0.5 * value_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_policy.parameters(), max_norm=1.0)
        optimizer.step()

        loss_history.append(float(total_loss.item()))
        rewards_history.append(float(raw_rewards.mean().item()))

    return {
        "final_ppo_loss": loss_history[-1],
        "final_penalized_reward": float(penalized_reward.mean().item()),
        "average_kl_divergence": float(kl_div.mean().item()),
        "loss_history": loss_history
    }
