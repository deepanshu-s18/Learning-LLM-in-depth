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
      6. Repeat until max_turns hit
    
    ⚠️  LIMITATION IN THIS BASELINE:
    There is NO warning to the model that time is running out.
    The model will often still be "exploring" when MAX_TURNS is hit.
    When there's no patch → reward = 0.
    This wastes rollouts and distorts the reward signal.
    (Mercor's "Context Nudge" fix addresses this — see 02_mercor_grpo.py)
    """
    env = MockEnv(fail_to_pass)
    context = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"ISSUE:\n{inst['problem_statement'][:1500]}"}
    ]

    for turn_idx in range(max_turns):
        # Build the prompt by applying the chat template
        prompt = tok.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
        
        # Let the model generate a response (this is the "action")
        reply  = sample(model, tok, prompt, temperature=temperature)
        
        # Parse the bash command out of the model's reply
        action = first_bash_block(reply)
        
        # Execute the command in the simulated environment
        obs = env.run(action) if action else NO_COMMAND
        
        # Append both sides to the context for next turn
        context += [
            {"role": "assistant", "content": reply},
            {"role": "user",      "content": obs[:800]}   # truncate long outputs
        ]
        
        # ⚠️  BASELINE GAP: No nudge injected here even when turn_idx == max_turns - 1
        # The model doesn't know it's the last turn.
        # Result: many rollouts end with reward = 0 because no patch was written.

    return dict(messages=context, patch=env.patch(), final=dict(env.fs), calls=env.calls)


# ─────────────────────────────────────────────────────────────────────
# STEP 4: THE REWARD FUNCTION (Verifier)
# ─────────────────────────────────────────────────────────────────────
def reward_random(patch, rng):
    """
    CONCEPT: The reward function answers "how good was this rollout?"
    
    In Mercor's system, the reward was computed by an LLM judge
    that read the final output and graded it against the task criteria.
    At 800 concurrent rollouts, this required round-robin API key rotation.
    
    Here we use a simple stand-in:
      - No patch written → reward = 0.0
      - Patch written → random score in [0, 1]
    
    In real training, you'd replace this with:
      - Unit test execution (SWE-bench)
      - LLM judge (Mercor APEX-Agents)
      - Triton kernel performance measurement (your triton-rl project)
    """
    if not patch:
        return 0.0
    return round(float(rng.random()), 3)


# ─────────────────────────────────────────────────────────────────────
# STEP 5: ADD LoRA ADAPTERS (trainable parameters)
# ─────────────────────────────────────────────────────────────────────
def add_lora(model):
    """
    CONCEPT: LoRA = Low-Rank Adaptation.
    
    We don't train ALL 500M parameters. That would:
    1. Require huge GPU memory
    2. Forget everything the model learned in pretraining
    
    Instead, we freeze the original weights and add tiny "adapter" 
    matrices (rank-8) to the attention layers. Only these adapters 
    are updated. This is called Parameter-Efficient Fine-Tuning (PEFT).
    
    In Mercor's 397B run, they used Megatron-LM with tensor parallelism
    across 8 training GPUs. We use PEFT LoRA on a single GPU.
    """
    model = get_peft_model(model, LoraConfig(
        r=8,           # rank of the adapter matrix (low = memory efficient)
        lora_alpha=16, # scaling factor (usually 2x rank)
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # attention layers
    ))
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────
# STEP 6: BUILD MASKED TOKEN IDS
# ─────────────────────────────────────────────────────────────────────
def build_masked(messages, tokenizer, max_len=3072):
    """
    CONCEPT: In RL, we only want to update the model on ASSISTANT tokens.
    The user/system tokens are the "observation" — we didn't generate them.
    
    We do this by:
    1. Tokenizing the full conversation
