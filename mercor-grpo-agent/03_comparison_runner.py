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
