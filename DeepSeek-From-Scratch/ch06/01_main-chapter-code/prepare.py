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

def encode_corpus(texts: List[str], enc: tiktoken.Encoding) -> np.ndarray:
    """Encodes a list of texts into a single flat stream of token IDs."""
    all_ids = []
    eot_token = enc.eot_token # End Of Text token
    for text in tqdm(texts, desc="Encoding texts"):
        # Encode each story and append the EOT token to act as a separator
        ids = enc.encode(text)
        all_ids.extend(ids)
        all_ids.append(eot_token)
    # Use uint16 since the vocab size is < 65535, saving disk space
    return np.array(all_ids, dtype=np.uint16)

def write_memmap(path: str, tokens: np.ndarray):
    """Writes a numpy array of tokens to a memory-mapped file."""
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(tokens.size,))
    arr[:] = tokens
    arr.flush()
    print(f"Wrote {tokens.size:,} tokens to {path}")

def main():
    """Main function to download, tokenize, and save the dataset."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load the dataset from Hugging Face
    print(f"Loading dataset: {DATASET_NAME}")
    # The dataset only has a 'train' split, which we will divide
    dataset = load_dataset(DATASET_NAME)['train']
    
    # 2. Create train and validation splits
    split_index = int(len(dataset) * (1 - VAL_RATIO))
    train_dataset = dataset.select(range(split_index))
    val_dataset = dataset.select(range(split_index, len(dataset)))
    
    print(f"Dataset split into {len(train_dataset):,} training and {len(val_dataset):,} validation samples.")

    # 3. Initialize the BPE tokenizer
    print(f"Initializing tokenizer...")
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    vocab_size = enc.n_vocab
    print(f"Tokenizer loaded. Vocab size: {vocab_size}")
    
    # 4. Tokenize the datasets
    train_ids = encode_corpus([ex['text'] for ex in train_dataset], enc)
    val_ids = encode_corpus([ex['text'] for ex in val_dataset], enc)

    # 5. Write the tokenized data to binary files for efficient loading
    write_memmap(os.path.join(OUTPUT_DIR, "train.bin"), train_ids)
    write_memmap(os.path.join(OUTPUT_DIR, "val.bin"), val_ids)
    
    # 6. Save metadata for the training script to use
    meta = {
        "tokenizer_name": TOKENIZER_NAME, # Store the reference name
        "vocab_size": vocab_size
    }
    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f)
    
    print(f"\nPreparation complete. Data is saved in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()