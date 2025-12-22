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
