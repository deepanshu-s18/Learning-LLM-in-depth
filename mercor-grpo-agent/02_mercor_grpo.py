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
