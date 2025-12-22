"""
Runner script for Process Reward Model (PRM) Guided Beam Search.
Executes step-level search-based reasoning and produces tree visualization.
"""

import argparse
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from src.prm_search import beam_search_with_prm, plot_trace_graph_tree_clean, get_default_device


def main():
    parser = argparse.ArgumentParser(description="Run PRM-Guided Beam Search for LLM Reasoning")
    parser.add_argument(
        "--reasoning-model",
        type=str,
        default="google/flan-t5-base",
        help="HF model ID for reasoning generation (e.g. google/flan-t5-base, TinyLlama/TinyLlama-1.1B-Chat-v1.0, HuggingFaceH4/zephyr-7b-beta)",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        help="HF model ID for PRM/Reward scoring (e.g. cross-encoder/ms-marco-MiniLM-L-12-v2 or OpenAssistant/reward-model-deberta-v3-large)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Roger has 5 tennis balls. He buys 2 cans of 3 tennis balls each. How many tennis balls does he have now?",
        help="Math / logic reasoning question",
    )
    parser.add_argument("--beams", type=int, default=4, help="Total beams (N)")
    parser.add_argument("--beam-width", type=int, default=2, help="Retained beams (M)")
    parser.add_argument("--max-steps", type=int, default=2, help="Number of search iterations")
    parser.add_argument("--output-plot", type=str, default="results/prm_beam_search_tree.png", help="Path to save tree plot")
    args = parser.parse_args()

    device = get_default_device()
    print("=" * 65, flush=True)
    print("🌳 PRM-Guided Beam Search (Inference-Time Compute Scaling)", flush=True)
