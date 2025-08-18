"""
Main entry point for running Few-Shot Chain of Thought Reasoning benchmarks.
"""

import argparse
import os
import json
from src.config import SEQ2SEQ_MODELS, DECODER_MODELS, MODEL_SIZES
from src.dataset import load_gsm8k, load_svamp
from src.prompts import FEW_SHOT_COT_PREFIX
from src.evaluator import Seq2SeqEvaluator, DecoderEvaluator
from src.visualize import plot_model_comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM Chain of Thought Reasoning Capabilities")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "svamp"], default="gsm8k", help="Dataset to evaluate on")
    parser.add_argument("--limit", type=int, default=50, help="Number of evaluation samples")
    parser.add_argument("--preview", type=int, default=3, help="Number of sample predictions to print")
    parser.add_argument("--models", nargs="+", default=["Flan-T5 Small", "Flan-T5 Base"], help="Models to evaluate")
    parser.add_argument("--output-plot", type=str, default="results/cot_reasoning_benchmark.png", help="Path to save chart")
