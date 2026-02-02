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
