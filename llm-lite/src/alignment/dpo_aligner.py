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
    loss = -F.logsigmoid(logits).mean()
    
    # Implicit reward margin
    implicit_margin = logits.detach().mean()
    return loss, implicit_margin


def train_dpo_alignment(
    policy_model: TransformerLM,
    preference_data: List[Dict[str, str]],
    tokenizer: SimpleTokenizer,
    epochs: int = 15,
    lr: float = 5e-4,
    beta: float = 0.1,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Executes Direct Preference Optimization.
    """
    policy_model.train()
    
    # Frozen Reference Policy pi_ref
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = CustomAdamW(policy_model.parameters(), lr=lr, weight_decay=0.0)
    block_size = policy_model.config.block_size

    # Prepare batches
    chosen_ids = []
    chosen_masks = []
    rejected_ids = []
    rejected_masks = []

    for item in preference_data:
        prompt_text = f"<|user|>{item['prompt']}<|assistant|>"
        c_text = f"{item['chosen']}<|end|>"
        r_text = f"{item['rejected']}<|end|>"

        p_enc = tokenizer.encode(prompt_text)
        c_enc = tokenizer.encode(c_text)
        r_enc = tokenizer.encode(r_text)

        # Chosen full seq
        full_c = (p_enc + c_enc)[:block_size]
        mask_c = [0] * (len(p_enc) - 1) + [1] * len(c_enc)
        mask_c = mask_c[: block_size - 1]
        
        # Pad
        pad_c_len = block_size - len(full_c)
        full_c += [tokenizer.pad_token_id] * pad_c_len
        mask_c += [0] * ((block_size - 1) - len(mask_c))

        # Rejected full seq
        full_r = (p_enc + r_enc)[:block_size]
        mask_r = [0] * (len(p_enc) - 1) + [1] * len(r_enc)
        mask_r = mask_r[: block_size - 1]

        pad_r_len = block_size - len(full_r)
        full_r += [tokenizer.pad_token_id] * pad_r_len
        mask_r += [0] * ((block_size - 1) - len(mask_r))

        chosen_ids.append(full_c)
        chosen_masks.append(mask_c)
        rejected_ids.append(full_r)
        rejected_masks.append(mask_r)

    c_ids_t = torch.tensor(chosen_ids, dtype=torch.long, device=device)
    c_mask_t = torch.tensor(chosen_masks, dtype=torch.float32, device=device)
    r_ids_t = torch.tensor(rejected_ids, dtype=torch.long, device=device)
    r_mask_t = torch.tensor(rejected_masks, dtype=torch.float32, device=device)

    # Precompute reference policy log probabilities once (frozen)
    with torch.no_grad():
        ref_logp_c = get_batch_log_probs(ref_model, c_ids_t, c_mask_t)
        ref_logp_r = get_batch_log_probs(ref_model, r_ids_t, r_mask_t)

    loss_history = []
    margin_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        pi_logp_c = get_batch_log_probs(policy_model, c_ids_t, c_mask_t)
        pi_logp_r = get_batch_log_probs(policy_model, r_ids_t, r_mask_t)

        loss, margin = compute_dpo_loss(pi_logp_c, pi_logp_r, ref_logp_c, ref_logp_r, beta=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_history.append(float(loss.item()))
        margin_history.append(float(margin.item()))

    return {
        "final_dpo_loss": loss_history[-1],
        "final_implicit_margin": margin_history[-1],
        "loss_history": loss_history
    }
