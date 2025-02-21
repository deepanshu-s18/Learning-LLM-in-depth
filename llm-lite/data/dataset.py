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
        self.user_token_id = 1
        self.assistant_token_id = 2
        self.end_token_id = 3

        # Vocab mapping: special tokens first, then characters
        self.id_to_token = list(self.special_tokens)
        for ch in chars:
            if ch not in self.id_to_token:
                self.id_to_token.append(ch)
                
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.id_to_token)}
        self.vocab_size = len(self.id_to_token)

    def encode(self, text: str) -> List[int]:
        """Encodes string text into integer token IDs, parsing special tokens."""
        tokens: List[int] = []
        i = 0
        n = len(text)
        while i < n:
            matched_special = False
            for st in self.special_tokens:
                if text[i:].startswith(st):
                    tokens.append(self.token_to_id[st])
                    i += len(st)
