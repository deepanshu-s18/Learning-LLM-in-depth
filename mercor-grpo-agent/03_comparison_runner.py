"""
03_comparison_runner.py
========================
Run both GRPO versions on the SAME rollouts and print a side-by-side
comparison showing exactly what changes and why.

This is educational — you will see:
  1. How much the token-length bias affects gradients (Fix #1)
  2. How many rollouts the context nudge rescues (Fix #2)
  3. How many tokens DPPO masks as "drifted" (Fix #3)

Run: python 03_comparison_runner.py
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
DPPO_DELTA = 0.2


def load_task():
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    cands = [i for i, r in enumerate(ds)
             if r["patch"].count("diff --git") == 1 and len(r["patch"]) < 1800]
    return ds[cands[0]], json.loads(ds[cands[0]]["FAIL_TO_PASS"])


def load_policy(device):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa"
    ).to(device)
    return model, tok


def add_lora(model):
    import copy
    import functools
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    return model


def build_masked(messages, tokenizer, max_len=3072):
    ids, labels, prev = [], [], ""
    for i, m in enumerate(messages):
        cur = tokenizer.apply_chat_template(messages[:i + 1], tokenize=False)
        assert cur.startswith(prev)
        seg = tokenizer(cur[len(prev):], add_special_tokens=False)["input_ids"]
        ids += seg
        labels += seg if m["role"] == "assistant" else [-100] * len(seg)
        prev = cur
    return ids[:max_len], labels[:max_len]


def seq_logprob_mean(model, tok, messages, device):
    """Baseline: token_mean."""
    ids, labs = build_masked(messages, tok)
    t = torch.tensor([ids], device=device)
    msk = torch.tensor([[0. if l == -100 else 1. for l in labs]], device=device)[:, 1:]
    logits = model(t).logits[:, :-1]
    lp = torch.log_softmax(logits.float(), -1).gather(-1, t[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (lp * msk).sum() / msk.sum().clamp(min=1), int(msk.sum().item())


def seq_logprob_per_token(model, tok, messages, device):
    """Mercor: per-token, needed for prompt_mean and DPPO."""
    ids, labs = build_masked(messages, tok)
    t = torch.tensor([ids], device=device)
    msk = torch.tensor([[0. if l == -100 else 1. for l in labs]], device=device)[:, 1:]
    logits = model(t).logits[:, :-1]
    lp = torch.log_softmax(logits.float(), -1).gather(-1, t[:, 1:].unsqueeze(-1)).squeeze(-1)
    return lp.squeeze(0), msk.squeeze(0), int(msk.sum().item())


def run_agent_baseline(model, tok, inst, fail_to_pass, rng):
    """Original harness — no nudge."""
    env = MockEnv(fail_to_pass)
    context = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"ISSUE:\n{inst['problem_statement'][:1500]}"}
    ]
    for _ in range(MAX_TURNS):
        prompt = tok.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
        reply  = generate(model, tok, prompt, temperature=TEMPERATURE)
        action = first_bash_block(reply)
        obs    = env.run(action) if action else NO_COMMAND
        context += [{"role": "assistant", "content": reply},
                    {"role": "user",      "content": obs[:800]}]
    patch = env.patch()
    score = round(float(rng.random()), 3) if patch else 0.0
    n_tok = sum(len(m["content"].split()) for m in context if m["role"] == "assistant")
    return dict(messages=context, patch=patch, reward=score, approx_tokens=n_tok)


def run_agent_nudged(model, tok, inst, fail_to_pass, rng):
    """Mercor harness — with context nudge (Fix #2)."""
    env = MockEnv(fail_to_pass)
    context = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"ISSUE:\n{inst['problem_statement'][:1500]}"}
    ]
    for turn_idx in range(MAX_TURNS):
        # ── [FIX #2] ────────────────────────────────────────────
        if turn_idx == MAX_TURNS - 1:
            context[-1]["content"] += (
                "\n\n⚠️ [SYSTEM]: Final turn — write your complete fix NOW."
            )
        # ────────────────────────────────────────────────────────
        prompt = tok.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
        reply  = generate(model, tok, prompt, temperature=TEMPERATURE)
        action = first_bash_block(reply)
        obs    = env.run(action) if action else NO_COMMAND
        context += [{"role": "assistant", "content": reply},
                    {"role": "user",      "content": obs[:800]}]
    patch = env.patch()
    score = round(float(rng.random()), 3) if patch else 0.0
    n_tok = sum(len(m["content"].split()) for m in context if m["role"] == "assistant")
    return dict(messages=context, patch=patch, reward=score, approx_tokens=n_tok)


def print_separator(title=""):
    print("\n" + "═" * 65)
    if title:
        print(f"  {title}")
        print("═" * 65)


def main():
    print_separator("MERCOR GRPO COMPARISON: BASELINE vs. UPGRADED")
    print("  Comparing 3 Mercor fixes on the same SWE-bench task.")
    print("  paper: mercor.com/blog/training-frontier-knowledge-work-agents...")
    
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    inst, fail_to_pass = load_task()
    print(f"Task: {inst['instance_id']}")

    model, tok = load_policy(device)
    rng_base = np.random.default_rng(SEED)
    rng_merc = np.random.default_rng(SEED)

    # ─────────────────────────────────────────────────────────────
    # COLLECT ROLLOUTS FROM BOTH HARNESSES
    # ─────────────────────────────────────────────────────────────
    print_separator("STEP 1: Collecting Rollouts")
    print(f"{'Rollout':>7}  {'Baseline Reward':>15}  {'Nudged Reward':>13}  "
          f"{'Baseline ~Tokens':>16}  {'Nudged ~Tokens':>14}")
    print("-" * 75)

    baseline_rollouts = []
    mercor_rollouts = []
    
    for i in range(GROUP_SIZE):
        b = run_agent_baseline(model, tok, inst, fail_to_pass, rng_base)
        m = run_agent_nudged(model, tok, inst, fail_to_pass, rng_merc)
        baseline_rollouts.append(b)
        mercor_rollouts.append(m)
        
        b_patch = "✅ YES" if b["patch"] else "❌ NO "
        m_patch = "✅ YES" if m["patch"] else "❌ NO "
        print(f"  {i:>5}  {b_patch} r={b['reward']:.3f}       "
              f"{m_patch} r={m['reward']:.3f}   "
              f"~{b['approx_tokens']:>5} tok         "
              f"~{m['approx_tokens']:>5} tok")

    b_zero = sum(1 for r in baseline_rollouts if r["reward"] == 0.0)
    m_zero = sum(1 for r in mercor_rollouts if r["reward"] == 0.0)

    # ─────────────────────────────────────────────────────────────
    # FIX #2 ANALYSIS
    # ─────────────────────────────────────────────────────────────
    print_separator("FIX #2 ANALYSIS: Context Nudge")
    print(f"  Zero-reward rollouts (baseline, no nudge):  {b_zero}/{GROUP_SIZE}")
    print(f"  Zero-reward rollouts (mercor, with nudge):  {m_zero}/{GROUP_SIZE}")
    rescued = max(0, b_zero - m_zero)
    if rescued > 0:
        print(f"\n  ✅ Context nudge RESCUED {rescued} rollout(s) from zero reward!")
        print(f"     These would have wasted GPU time in the baseline.")
    else:
        print(f"\n  ℹ️  No rescue in this run — the model happened to write patches anyway.")
        print(f"     On longer tasks (Mercor's 50+ turn tasks), the nudge gives +3.0 pts.")

    # ─────────────────────────────────────────────────────────────
    # FIX #1 ANALYSIS
    # ─────────────────────────────────────────────────────────────
    print_separator("FIX #1 ANALYSIS: prompt_mean vs token_mean")
    
    b_rewards = np.array([r["reward"] for r in baseline_rollouts])
    b_tokens  = np.array([r["approx_tokens"] for r in baseline_rollouts])
    b_adv     = (b_rewards - b_rewards.mean()) / (b_rewards.std() + 1e-4)
    
    print(f"\n  {'Rollout':>7}  {'Reward':>8}  {'Advantage':>10}  "
          f"{'~Tokens':>8}  {'token_mean weight':>18}  {'prompt_mean weight':>19}")
    print("  " + "-" * 80)
    
    total_tokens = b_tokens.sum()
    
    for i, (r, a, t) in enumerate(zip(b_rewards, b_adv, b_tokens)):
        token_weight   = t / total_tokens * 100    # % of gradient (token_mean)
        prompt_weight  = 100.0 / GROUP_SIZE         # % of gradient (prompt_mean — always equal)
        flag = " ← BIASED" if token_weight > prompt_weight * 1.5 else ""
        print(f"  {i:>7}  {r:>8.3f}  {a:>10.3f}  {t:>8}  "
              f"{token_weight:>17.1f}%  {prompt_weight:>18.1f}%{flag}")
    
    max_bias = b_tokens.max() / b_tokens.min() if b_tokens.min() > 0 else float("inf")
    print(f"\n  token_mean: longest rollout contributes {max_bias:.1f}x MORE than shortest ⚠️")
    print(f"  prompt_mean: every rollout contributes equally (1/{GROUP_SIZE} = {100/GROUP_SIZE:.1f}%) ✅")

    # ─────────────────────────────────────────────────────────────
    # FIX #3 ANALYSIS
    # ─────────────────────────────────────────────────────────────
    print_separator("FIX #3 ANALYSIS: DPPO Token Masking")
    print(f"  Checking how many tokens would be masked at delta={DPPO_DELTA}")
    print(f"  (In a real training run, LoRA adapters update across many mini-batches,")
    print(f"   so policy drift accumulates. We simulate with small random noise.)\n")
    
    with torch.no_grad():
        for i, rollout in enumerate(mercor_rollouts):
            lp_old, msk, n_tok = seq_logprob_per_token(model, tok, rollout["messages"], device)
            if n_tok == 0:
                continue
            # Simulate a small policy drift (as if one training step happened)
            drift = torch.randn_like(lp_old) * 0.05
            lp_drifted = lp_old + drift
            ratio = (lp_drifted - lp_old).exp()
            tv = (ratio - 1.0).abs()
            masked_pct = (tv[msk.bool()] > DPPO_DELTA).float().mean().item() * 100
            print(f"  rollout {i}: {n_tok} assistant tokens — "
                  f"DPPO would mask {masked_pct:.1f}% as 'policy drifted'")
    
    print(f"\n  ✅ DPPO ensures only 'stable' tokens affect the gradient.")
    print(f"     This prevents catastrophic policy collapse on long agent trajectories.")

    # ─────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────
    print_separator("SUMMARY: What the Mercor Paper Did")
    print("""
  PROBLEM: Standard GRPO fails for multi-turn agents because:
    1. Long rambling rollouts dominate gradients (token_mean bias)
    2. Agents hit turn limits and get zero reward (no warning)
    3. Policy drift on multi-turn sequences causes instability

  MERCOR'S 3 FIXES (add up to +7-8 points on APEX-Agents):

    Fix #1 [prompt_mean]:   +3.9 pts
      Every rollout contributes equally to the gradient.
      Short efficient rollouts aren't drowned out by long rambling ones.

    Fix #2 [context nudge]: +3.0 pts
      The agent is warned on the last turn to commit to its answer.
      This converts zero-reward "ran-out-of-turns" rollouts into
      actual scored rollouts.

    Fix #3 [DPPO masking]:  stability improvement (not a direct pts gain)
      Token-level importance masking prevents policy collapse
      on async/multi-step training. Mercor observed: more deliberate
      behavior (more turns, fewer tokens per turn).

  BIG LESSON FROM THE PAPER:
    "Algorithm choices mattered less than the data: the best of
     five knobs gave +3.9 points, while post-training as a whole
     moved both models 10 to 12 points."

    → The harness fixes (Fix #2) and data quality matter MORE than
      fancy algorithm choices. Fix boring infrastructure first.
""")


if __name__ == "__main__":
    main()
