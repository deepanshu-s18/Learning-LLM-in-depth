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
