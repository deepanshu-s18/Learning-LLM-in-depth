"""
interactive_chat.py
Interactive CLI Playground for LLM-Lite.
Allows live prompt testing across all trained model checkpoints (Base, SFT, PPO, DPO),
with toggleable KV-Cache acceleration and real-time generation metrics.
"""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import SimpleTokenizer
from src.model.transformer import TransformerLM, TransformerConfig
from src.foundations.pytorch_engine import get_device, load_checkpoint
from src.tuning.lora import inject_lora

def run_chat_studio():
    device = get_device()
