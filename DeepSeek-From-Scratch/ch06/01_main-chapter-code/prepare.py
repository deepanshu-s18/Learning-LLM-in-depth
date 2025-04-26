# prepare.py
# This script downloads the TinyStories dataset, tokenizes it using a standard
# Byte-Pair Encoding (BPE) tokenizer, and saves the data for training.

import os
import json
from typing import List

import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm import tqdm # For progress bars

# --- Configuration ---
DATASET_NAME = "roneneldan/TinyStories"
# We use a standard BPE tokenizer with a ~50k vocabulary size.
# 'gpt2' is the reference name for this tokenizer in the tiktoken library.
TOKENIZER_NAME = "gpt2"
OUTPUT_DIR = "data/tinystories_tokenized" # Generic output directory name
VAL_RATIO = 0.05
