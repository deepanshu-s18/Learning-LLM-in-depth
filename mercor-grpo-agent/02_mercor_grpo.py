"""
02_mercor_grpo.py
==================
THE UPGRADE: Mercor-Style GRPO with 3 Surgical Fixes

Every change from the baseline is marked with:
    [MERCOR FIX #1] — prompt_mean token aggregation
    [MERCOR FIX #2] — context nudge harness change
    [MERCOR FIX #3] — DPPO divergence masking

Source paper: "Training Frontier Knowledge Work Agents: A 397B RL Training Guide with SkyRL"
              Mercor Research + SkyRL (Berkeley), September 2026
              https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/

Run: python 02_mercor_grpo.py
"""

import json
import os
import random
import urllib.request
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

if not os.path.exists("utils.py"):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/abgoswam/swe_in_prod_vizuara_01/main/utils.py",
        "utils.py")

from utils import NO_COMMAND, SYSTEM, MockEnv, first_bash_block, generate

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
GROUP_SIZE = 6
MAX_TURNS = 4
TEMPERATURE = 1.0
LR = 1e-5
SEED = 0

# ─────────────────────────────────────────────────────────────────────
# [MERCOR FIX #3 SETUP] — DPPO Hyperparameter
# ─────────────────────────────────────────────────────────────────────
DPPO_DELTA = 0.2
# ^ The threshold for masking out policy-diverged tokens.
# If |ratio - 1| > DPPO_DELTA, that token is masked from the loss.
# Mercor uses total-variation divergence approximation.
# 0.2 means: if the current policy is more than 20% away from the
# rollout policy on any token, ignore that token's gradient.


# ─────────────────────────────────────────────────────────────────────
# LOAD TASK AND POLICY (unchanged from baseline)
# ─────────────────────────────────────────────────────────────────────
def load_task():
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    cands = [i for i, r in enumerate(ds)
             if r["patch"].count("diff --git") == 1 and len(r["patch"]) < 1800]
    inst = ds[cands[0]]
    return inst, json.loads(inst["FAIL_TO_PASS"])


def load_policy(device):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa"
    ).to(device)
    print(f"{MODEL}\n{sum(p.numel() for p in model.parameters()):,} parameters")
    return model, tok


def add_lora(model):
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────
# [MERCOR FIX #2] — CONTEXT NUDGE HARNESS
# ─────────────────────────────────────────────────────────────────────
def run_agent_with_nudge(model, tok, inst, fail_to_pass, max_turns=MAX_TURNS,
                         temperature=TEMPERATURE, sample=generate):
    """
    ╔═══════════════════════════════════════════════════════════════╗
    ║  [MERCOR FIX #2]: CONTEXT NUDGE                              ║
    ║                                                               ║
    ║  What changed vs baseline:                                    ║
    ║    - When turn == max_turns - 1 (second-to-last turn),        ║
    ║      we inject a WARNING into the user message telling the    ║
    ║      model to stop exploring and commit to a final answer.    ║
    ║                                                               ║
    ║  Why this works:                                              ║
    ║    - In the baseline, the model doesn't know it's about to   ║
    ║      hit MAX_TURNS. So it keeps exploring and never writes    ║
    ║      the final patch.                                         ║
    ║    - Result: many rollouts end with reward = 0.0 (no patch)  ║
    ║    - These zero-reward rollouts still consume GPU time and    ║
    ║      bias the advantage calculation.                          ║
    ║                                                               ║
    ║  Mercor result: +3.0 points with ZERO training cost.          ║
    ║  Quote from paper: "Fewer rollouts blow the context, so       ║
    ║  fewer get zeroed across the board, hence more usable        ║
    ║  signal per batch."                                           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    env = MockEnv(fail_to_pass)
    context = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"ISSUE:\n{inst['problem_statement'][:1500]}"}
    ]

    for turn_idx in range(max_turns):
        
        # ─────────────────────────────────────────────────────────
        # [MERCOR FIX #2]: INJECT THE CONTEXT NUDGE
        # When we're at 80% of the turn budget (second-to-last turn),
        # append a warning to the LATEST user message.
        # ─────────────────────────────────────────────────────────
        if turn_idx == max_turns - 1:
            # Modify the last user turn to include the warning
            nudge_text = (
                "\n\n⚠️  [SYSTEM ALERT]: This is your FINAL turn. "
                "You MUST stop exploring and write your complete fix now. "
                "Produce the full patch immediately."
            )
            # Append nudge to the last message in context
            context[-1]["content"] = context[-1]["content"] + nudge_text
        
        prompt = tok.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
        reply  = sample(model, tok, prompt, temperature=temperature)
        action = first_bash_block(reply)
        obs    = env.run(action) if action else NO_COMMAND
        context += [
            {"role": "assistant", "content": reply},
            {"role": "user",      "content": obs[:800]}
        ]

    return dict(messages=context, patch=env.patch(), final=dict(env.fs), calls=env.calls)


def reward_random(patch, rng):
    if not patch:
        return 0.0
    return round(float(rng.random()), 3)


# ─────────────────────────────────────────────────────────────────────
# TOKEN-LEVEL LOG PROB (needed for Fix #1 and Fix #3)
# ─────────────────────────────────────────────────────────────────────
def build_masked(messages, tokenizer, max_len=3072):
    """Identical to baseline — tokenize and mask user/system tokens."""
    ids, labels, prev = [], [], ""
    for i, m in enumerate(messages):
        cur = tokenizer.apply_chat_template(messages[:i + 1], tokenize=False)
        assert cur.startswith(prev)
        seg = tokenizer(cur[len(prev):], add_special_tokens=False)["input_ids"]
        ids += seg
        labels += seg if m["role"] == "assistant" else [-100] * len(seg)
        prev = cur
    return ids[:max_len], labels[:max_len]


def seq_logprob_per_token(model, tok, messages, device):
    """
    ╔═══════════════════════════════════════════════════════════════╗
    ║  NEW FUNCTION: Returns PER-TOKEN logprobs, not the mean.     ║
    ║                                                               ║
    ║  This is needed for both Fix #1 and Fix #3.                  ║
    ║  - Fix #1 needs to normalize per-rollout (not globally)      ║
    ║  - Fix #3 needs token-level values for divergence masking    ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Returns:
        per_token_lp: tensor of shape [seq_len-1] — logprob per position
        mask:         tensor of shape [seq_len-1] — 1 for assistant tokens
        n_tokens:     int — total supervised tokens
    """
    ids, labs = build_masked(messages, tok)
    t = torch.tensor([ids], device=device)
    msk = torch.tensor([[0. if l == -100 else 1. for l in labs]], device=device)[:, 1:]
    logits = model(t).logits[:, :-1]
    per_token_lp = torch.log_softmax(logits.float(), -1).gather(
        -1, t[:, 1:].unsqueeze(-1)
    ).squeeze(-1)   # shape: [1, seq_len-1]
    
    return per_token_lp.squeeze(0), msk.squeeze(0), int(msk.sum().item())


# ─────────────────────────────────────────────────────────────────────
# [MERCOR FIX #1 + #3] — MERCOR-STYLE LOSS COMPUTATION
# ─────────────────────────────────────────────────────────────────────
def compute_mercor_loss(model, tok, group, adv, rollout_logprobs, device,
                        delta=DPPO_DELTA):
    """
    ╔═══════════════════════════════════════════════════════════════╗
    ║  [MERCOR FIX #1]: PROMPT_MEAN TOKEN AGGREGATION             ║
    ║                                                               ║
    ║  THE PROBLEM with token_mean (baseline):                     ║
    ║    - Rollout A: 200 tokens, reward=0.85 (efficient!)         ║
    ║    - Rollout B: 5000 tokens, reward=0.0  (rambling!)         ║
    ║    - token_mean gives 25x more weight to rollout B           ║
    ║      because it has 25x more tokens.                         ║
    ║    - The gradient is dominated by the bad rollout. ❌        ║
    ║                                                               ║
    ║  THE FIX — prompt_mean:                                       ║
    ║    - Normalize each rollout by ITS OWN token count first      ║
    ║    - THEN average across the group                           ║
    ║    - Every rollout contributes EQUALLY to the gradient ✅     ║
    ║                                                               ║
    ║  Formula:                                                     ║
    ║    L = -1/G * Σᵢ (Aᵢ * Σₜ logπ(xₜ|x<t) / Tᵢ)              ║
    ║    where Tᵢ = number of assistant tokens in rollout i         ║
    ║                                                               ║
    ║  Mercor result: +3.9 points improvement.                      ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    ╔═══════════════════════════════════════════════════════════════╗
    ║  [MERCOR FIX #3]: DPPO DIVERGENCE MASKING                   ║
    ║                                                               ║
    ║  THE PROBLEM without DPPO:                                   ║
    ║    - The rollout was collected using policy π_old             ║
    ║    - After updates, current policy π_θ has drifted           ║
    ║    - Applying gradients on stale rollouts → instability      ║
    ║                                                               ║
    ║  THE FIX — DPPO (from arXiv:2602.04879):                    ║
    ║    1. Compute importance ratio r_t = exp(π_θ / π_old)        ║
    ║    2. For each token, check if |r_t - 1| > delta             ║
    ║    3. If YES → mask this token out (don't include in loss)   ║
    ║    4. Only update on tokens where policy is still "close"    ║
    ║                                                               ║
    ║  Effect observed by Mercor: more deliberate behavior         ║
    ║  (more turns, fewer tokens per turn = focused exploration)   ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Args:
        rollout_logprobs: list of per-token logprob tensors collected
                          BEFORE the current update step (the "old" policy).
                          These are frozen snapshots used for Fix #3.
    """
    total_loss = 0.0
    group_size = len(group)
    
    for i, (g, a, old_lp_per_tok) in enumerate(zip(group, adv, rollout_logprobs)):
        
        # ── Get per-token logprobs under CURRENT policy ──────────
        curr_lp_per_tok, mask, n_tokens = seq_logprob_per_token(
            model, tok, g["messages"], device
        )
        
        if n_tokens == 0:
            continue   # no assistant tokens in this rollout, skip
        
        # ─────────────────────────────────────────────────────────
        # [MERCOR FIX #3]: COMPUTE DPPO DIVERGENCE MASK
        # ─────────────────────────────────────────────────────────
        # Move old logprobs to device for comparison
        old_lp_per_tok = old_lp_per_tok.to(device)
        
        # Importance ratio: how much has the policy changed at each token?
        # r_t = exp(log π_θ(x_t) - log π_old(x_t))
        # Since we're in log space: ratio = exp(current - old)
        ratio = (curr_lp_per_tok - old_lp_per_tok).exp()
        
        # Total variation approximation: |r_t - 1| is a proxy for divergence
        tv_divergence = (ratio - 1.0).abs()
        
        # DPPO mask: keep only tokens where policy hasn't drifted too far
        # True = stable token (include in loss), False = drifted (exclude)
        dppo_mask = (tv_divergence < delta).float()
        
        # Combine with supervised token mask (only assistant tokens)
        combined_mask = mask * dppo_mask
        
        # Count stable assistant tokens
        n_stable_tokens = combined_mask.sum()
        
        if n_stable_tokens == 0:
            # All tokens drifted too far — skip this rollout entirely
            print(f"    rollout {i}: ALL tokens masked by DPPO (policy drifted > {delta})")
            continue
        
        # ─────────────────────────────────────────────────────────
        # [MERCOR FIX #1]: PROMPT_MEAN — normalize by THIS rollout's tokens
        # ─────────────────────────────────────────────────────────
        # Sum logprobs only over stable assistant tokens
        rollout_lp_sum = (curr_lp_per_tok * combined_mask).sum()
        
        # Normalize by number of stable tokens in THIS rollout (not global!)
        # This is what makes it "prompt_mean" instead of "token_mean"
        rollout_lp_mean = rollout_lp_sum / n_stable_tokens
        
        # REINFORCE loss: negative because we maximize reward
        # Divide by group_size to average across the group (the "prompt mean" part)
        loss = -(a.to(device) * rollout_lp_mean) / group_size
        loss.backward()
        total_loss += loss.item()
        
        # Debug info
        pct_masked = 100 * (1 - dppo_mask.mean().item())
        print(f"    rollout {i}: advantage={a:.3f}  tokens={n_tokens}"
              f"  DPPO masked={pct_masked:.0f}%"
              f"  stable={int(n_stable_tokens.item())}")
    
    return total_loss


# ─────────────────────────────────────────────────────────────────────
# MERCOR-STYLE GRPO UPDATE (combines all 3 fixes)
# ─────────────────────────────────────────────────────────────────────
def grpo_step_mercor(model, tok, group, reward, rollout_logprobs, device, lr=LR):
    """
    Full Mercor-style GRPO step with all 3 fixes applied.
    
    This is what their SkyRL runner does for each mini-batch:
    1. Compute advantages (same as baseline)
    2. Apply prompt_mean loss (Fix #1)
    3. Apply DPPO masking (Fix #3)
    (Fix #2 was applied during rollout collection in run_agent_with_nudge)
    """
    rewards = torch.tensor(reward, dtype=torch.float)
    adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
    
    print(f"\n--- GRPO STEP (MERCOR UPGRADED) ---")
    print(f"mean reward {rewards.mean():.3f}   std {rewards.std(unbiased=False):.3f}")
    
    for i, (r, a) in enumerate(zip(rewards, adv)):
        print(f"  rollout {i}: reward={r:.3f}  advantage={a:.3f}")
    
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    
    # Log logprobs before update
    logp_before = []
    for g in group:
        lp_tok, msk, _ = seq_logprob_per_token(model, tok, g["messages"], device)
        n = msk.sum()
        logp_before.append((lp_tok * msk).sum().item() / n.clamp(min=1).item())
    
    model.train()
    opt.zero_grad(set_to_none=True)
    
    print(f"\nApplying Mercor fixes [#1 prompt_mean + #3 DPPO delta={DPPO_DELTA}]:")
    
    # ── MERCOR LOSS (Fix #1 + Fix #3) ───────────────────────────
    loss_total = compute_mercor_loss(model, tok, group, adv, rollout_logprobs, device)
    
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    
    # Log logprobs after update
    logp_after = []
    for g in group:
        lp_tok, msk, _ = seq_logprob_per_token(model, tok, g["messages"], device)
        n = msk.sum()
        logp_after.append((lp_tok * msk).sum().item() / n.clamp(min=1).item())
    
    print(f"\nloss {loss_total:+.5f}   grad_norm {grad_norm:.4f}")
    print("\nLogprob changes after Mercor update:")
    for i, (a, b, c) in enumerate(zip(adv, logp_before, logp_after)):
        direction = "↑" if c > b else "↓"
        print(f"  rollout {i}: advantage={a:.3f}  {b:.4f} → {c:.4f}  {direction}  (Δ={c-b:.5f})")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  MERCOR-UPGRADED GRPO (3 Fixes Applied)")
    print("  Fix #1: prompt_mean (+3.9 pts in paper)")
    print("  Fix #2: context nudge (+3.0 pts in paper)")
    print("  Fix #3: DPPO token divergence masking")
    print("=" * 65)
    
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. Load task
    inst, fail_to_pass = load_task()
    print(f"\nTask: {inst['instance_id']}  |  {len(fail_to_pass)} tests must go FAIL → PASS")

    # 2. Load policy (no LoRA yet)
    model, tok = load_policy(device)

    # 3. Collect rollouts using the NUDGED harness [Fix #2]
    rng = np.random.default_rng(SEED)
    group, reward = [], []
    
    print(f"\nCollecting {GROUP_SIZE} rollouts with CONTEXT NUDGE (Fix #2)...")
    for i in range(GROUP_SIZE):
        rollout = run_agent_with_nudge(model, tok, inst, fail_to_pass)
        score = reward_random(rollout["patch"], rng)
        group.append(rollout)
        reward.append(score)
        print(f"  rollout {i}: patch={'YES' if rollout['patch'] else 'NO'}  reward={score}")

    reward = np.array(reward)
    
    # 4. Snapshot rollout logprobs BEFORE adding LoRA [needed for Fix #3 DPPO]
    # We need the OLD policy's logprobs to compute divergence ratios.
    print(f"\nSnapshotting rollout logprobs (for DPPO Fix #3)...")
    rollout_logprobs = []
    with torch.no_grad():
        for g in group:
            lp_tok, _, _ = seq_logprob_per_token(model, tok, g["messages"], device)
            rollout_logprobs.append(lp_tok.detach().cpu())   # store on CPU to save VRAM

    # 5. Add LoRA adapters
    print(f"\nAdding LoRA adapters for training...")
    model = add_lora(model)

    # 6. Mercor GRPO step [Fix #1 + #3]
    grpo_step_mercor(model, tok, group, reward, rollout_logprobs, device)

    print("\n✅ Mercor-upgraded GRPO complete.")
    print("   Run 03_comparison_runner.py to compare baseline vs. Mercor side-by-side.")


if __name__ == "__main__":
    main()
