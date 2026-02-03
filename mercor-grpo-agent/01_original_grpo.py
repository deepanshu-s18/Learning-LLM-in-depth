"""
01_original_grpo.py
====================
THE BASELINE: Original GRPO Agent (unchanged logic, heavily annotated)

This is the exact code from swe_in_prod_vizuara_01/swe_grpo_one_step.py
with comments added to EVERY important line explaining what it does
and WHY it has limitations for multi-turn agents.

Run: python 01_original_grpo.py
"""

import json
import os
import random
import urllib.request
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────
# WHAT IS SWE-bench?
# SWE-bench is a dataset of real GitHub issues from popular Python repos
# (like Django, Flask, Numpy). Each task has:
#   - A problem_statement: the GitHub issue text
#   - A patch: the correct code fix
#   - FAIL_TO_PASS: tests that must go from failing to passing
#
# The agent must read the issue, explore the code, and write a fix.
# This is called a "software engineering agent task".
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
# ^ We use Qwen2.5-Coder-0.5B — a tiny 500M parameter coding model.
# Mercor used 397B parameters. We're doing the same math on a tiny scale
# so you can run it on any laptop.

GROUP_SIZE = 6
# ^ GRPO's key idea: instead of one rollout, run GROUP_SIZE rollouts
# of the SAME task. Compare them against each other. The better ones
# get positive advantage, the worse ones get negative advantage.

MAX_TURNS = 4
# ^ Each rollout = 4 turns (4 times the model can think + act).
# In Mercor's system, this was up to 100+ turns on complex tasks.

TEMPERATURE = 1.0
# ^ Higher temperature = more random = more diverse rollouts.
# We want diversity so different rollouts get different rewards.

LR = 1e-5
# ^ Learning rate for the gradient update.

SEED = 0

# ─────────────────────────────────────────────────────────────────────
# DOWNLOAD UTILITY FUNCTIONS
# (MockEnv, generate, first_bash_block, SYSTEM prompt — from the original repo)
# ─────────────────────────────────────────────────────────────────────
if not os.path.exists("utils.py"):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/abgoswam/swe_in_prod_vizuara_01/main/utils.py",
        "utils.py")

from utils import NO_COMMAND, SYSTEM, MockEnv, first_bash_block, generate


# ─────────────────────────────────────────────────────────────────────
# STEP 1: LOAD THE TASK
# ─────────────────────────────────────────────────────────────────────
def load_task():
    """
    Loads one task from SWE-bench_Verified.
    
    We pick the simplest task: one file changed, short patch.
    Real training would loop over thousands of these tasks.
    
    Returns:
        inst: A dict with keys: problem_statement, patch, instance_id, etc.
        fail_to_pass: List of test names that must go from FAIL → PASS.
    """
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    # Filter to tasks with only 1 file changed and a short patch (easier)
    cands = [i for i, r in enumerate(ds)
             if r["patch"].count("diff --git") == 1 and len(r["patch"]) < 1800]
    inst = ds[cands[0]]
    return inst, json.loads(inst["FAIL_TO_PASS"])


# ─────────────────────────────────────────────────────────────────────
# STEP 2: LOAD THE POLICY (the model)
# ─────────────────────────────────────────────────────────────────────
def load_policy(device):
    """
    CONCEPT: In RL, the "policy" is the brain that makes decisions.
    Here, the policy is the LLM (Qwen2.5-Coder-0.5B).
    
    We load it WITHOUT LoRA adapters yet. We don't add trainable
    parameters until AFTER we collect all rollouts. This is because:
    - Generation (rollouts) needs fast inference → no extra parameters
    - Training needs gradients → add LoRA adapters just before update
    
    This separation is called "actor-critic decoupling" and is standard
    in production RL (also how SkyRL/vLLM separate rollout from training).
    """
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token   # some models need this

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa"   # scaled dot product attention (fast)
    ).to(device)

    print(f"{MODEL}\n{sum(p.numel() for p in model.parameters()):,} parameters, none trainable yet")
    return model, tok


# ─────────────────────────────────────────────────────────────────────
# STEP 3: RUN THE AGENT (collect one rollout/trajectory)
# ─────────────────────────────────────────────────────────────────────
def run_agent(model, tok, inst, fail_to_pass, max_turns=MAX_TURNS,
              temperature=TEMPERATURE, sample=generate):
    """
    CONCEPT: The "harness" = the environment the agent acts in.
    
    In Mercor's system, the harness was a Docker container with:
      - A simulated company file system
      - PDF/Excel/Slides MCP servers
      - Email and Slack servers
    
    Here, the harness is MockEnv — a simple simulated bash terminal.
    
    THE AGENT LOOP (ReAct pattern):
    For each turn:
      1. Show model the current context (system prompt + issue + history)
      2. Model generates a reply (thinks + writes bash command)
      3. We extract the bash command from the reply
      4. Run the command in MockEnv (fake terminal)
      5. Append model output + terminal result to context
