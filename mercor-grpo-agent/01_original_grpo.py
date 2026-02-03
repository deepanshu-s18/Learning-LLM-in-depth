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
