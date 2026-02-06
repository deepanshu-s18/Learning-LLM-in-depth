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
