"""
data/dataset.py
Unified Tokenizer and Dataset Generators for Pretraining, SFT, and Preference Alignment (DPO/RLHF).
"""

import torch
from typing import List, Tuple, Dict

SPECIAL_TOKENS = ["<|pad|>", "<|user|>", "<|assistant|>", "<|end|>"]

class SimpleTokenizer:
    """
    Lightweight character-level tokenizer with support for custom special tokens.
    Provides fast, deterministic encoding/decoding for experimentation without external dependencies.
    """
    def __init__(self, extra_text: str = ""):
        chars = sorted(list(set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-:;'\n\"()[]{}/*+=$#@%&<>|"
            + extra_text
        )))
        
        self.special_tokens = SPECIAL_TOKENS
        self.pad_token_id = 0
