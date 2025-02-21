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
                    matched_special = True
                    break
            if not matched_special:
                ch = text[i]
                tokens.append(self.token_to_id.get(ch, self.token_to_id.get("?", 0)))
                i += 1
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decodes list of token IDs back into string text."""
        result = []
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                result.append(self.id_to_token[tid])
        return "".join(result)


def get_pretrain_corpus() -> List[str]:
    """
    Curated text corpus for causal pretraining.
    Contains diverse sentences covering concepts, instructions, code, and dialogue.
    """
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming science, engineering, and society.",
        "Deep neural networks learn representations through gradient descent and backpropagation.",
        "The transformer architecture uses self-attention mechanisms to model relationships.",
        "Rotary positional embeddings encode relative distances between tokens in high-dimensional space.",
        "Key-value caching accelerates autoregressive decoding by saving previous token computations.",
        "Low-rank adaptation freezes base model weights and trains low-rank matrix pairs.",
        "Reinforcement learning from human feedback aligns language models with human intentions.",
        "Direct preference optimization mathematically optimizes policy probabilities without a separate reward model.",
        "Quantization compresses 32-bit floating point weights into 8-bit and 4-bit integers.",
        "Python is a versatile programming language widely used in machine learning and data science.",
        "To write clean code, write modular functions with descriptive names and comprehensive tests.",
        "Language models predict the next token given the context of previous tokens.",
        "Knowledge retrieval augmented generation grounds language model responses in verified facts.",
        "Optimizers like Adam combine momentum and root mean square propagation with bias corrections.",
        "The loss function measures the discrepancy between predicted logits and true target distributions."
    ]


def get_sft_dataset() -> List[Dict[str, str]]:
    """
    Supervised Fine-Tuning dataset with (prompt, response) pairs.
    """
    return [
        {
            "prompt": "What is backpropagation?",
            "response": "Backpropagation computes gradients of the loss with respect to parameters using the chain rule."
        },
        {
            "prompt": "Why use the Adam optimizer?",
            "response": "Adam adapts per-parameter learning rates using exponentially decaying averages of past gradients and squared gradients."
        },
        {
            "prompt": "How does multi-head attention work?",
            "response": "It projects queries, keys, and values into multiple subspaces, calculates scaled dot-product attention in parallel, and concatenates the outputs."
        },
        {
            "prompt": "What is KV caching in LLMs?",
            "response": "KV caching stores past key and value projections so autoregressive generation does not recompute previous tokens at each step."
        },
        {
            "prompt": "Explain RoPE positional encoding.",
            "response": "RoPE applies 2D rotation matrices to query and key vectors so that attention scores depend on the relative distance between positions."
        },
        {
            "prompt": "What is LoRA parameter-efficient fine-tuning?",
            "response": "LoRA freezes the pretrained weights and adds trainable low-rank decomposition matrices to linear layers."
        },
        {
            "prompt": "What is the difference between DPO and PPO?",
            "response": "PPO trains a policy using a separate reward model and reinforcement learning, while DPO optimizes the policy directly from preference pairs."
        },
        {
            "prompt": "How does model quantization reduce memory?",
            "response": "Quantization maps 32-bit floating-point weights into 8-bit or 4-bit integers using scale and zero-point calibration."
        }
    ]


def get_preference_dataset() -> List[Dict[str, str]]:
    """
    Preference dataset with (prompt, chosen, rejected) triples for DPO and Reward Model training.
    """
    return [
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris.",
            "rejected": "France is a country in Europe with cities and mountains."
        },
        {
            "prompt": "How do you define a function in Python?",
            "chosen": "You define a function in Python using the def keyword followed by the function name and parentheses.",
            "rejected": "function myFunc() { return true; }"
        },
        {
            "prompt": "Explain what a transformer is in AI.",
            "chosen": "A transformer is a deep learning architecture that relies on self-attention to process entire sequences in parallel.",
            "rejected": "Transformers are giant robots from outer space that turn into cars and trucks."
        },
        {
            "prompt": "What is the benefit of LoRA fine-tuning?",
            "chosen": "LoRA reduces trainable parameters by over 90%, lowering GPU memory requirements while preserving model quality.",
            "rejected": "LoRA makes the model 100 times larger and requires thousands of expensive GPUs."
        },
        {
            "prompt": "Why is KV cache important during LLM generation?",
            "chosen": "It avoids quadratic redundant computations, speeding up token generation by 5x to 10x.",
            "rejected": "KV cache slows down the model by deleting previous words from memory."
        },
        {
            "prompt": "What is Direct Preference Optimization (DPO)?",
            "chosen": "DPO directly optimizes policy weights using human preference pairs without needing an explicit reward model or RL loop.",
            "rejected": "DPO is a database optimization algorithm for indexing SQL tables."
        }
    ]
