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
