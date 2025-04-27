# train.py
# This script trains the MiniDeepSeek model on the tokenized TinyStories dataset.
# It is designed to run on a single GPU.

import os
import json
import time
from contextlib import nullcontext
import math # Added for learning rate scheduler

import numpy as np
import torch
from torch.nn import functional as F

# Import the model definition from model.py
from model import MiniDeepSeek, ModelArgs

# --- Configuration ---
# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device.split(':')[0], dtype=pt_dtype) if 'cuda' in device else nullcontext()

# Training
out_dir = 'out'
data_dir = 'data/tinystories_tokenized' # Points to the generic output directory
max_iters = 5000
eval_interval = 250          # Evaluate less frequently on a longer run
log_interval = 20
eval_iters = 100
batch_size = 24              # A good starting batch size for a 4090
block_size = 256

# AdamW Optimizer
learning_rate = 4e-4         # A slightly higher LR can work well for this size
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95

# --- Data Loading ---
def get_batch(split: str):
    """
    Loads a batch of data from the memory-mapped .bin files.
    """
    # The data files are now uint16, as created by the new prepare.py
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    # Generate random starting points for each sequence in the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Create input sequences (x)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    # Create target sequences (y), which are shifted by one
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if 'cuda' in device:
        # Pin memory helps speed up CPU-to-GPU data transfer
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        return x, y

# Learning rate scheduler: cosine decay with warmup
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    warmup_iters = 200 # A bit longer warmup for a longer run
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min_lr
    lr_decay_iters = max_iters
    min_lr = learning_rate / 10
    if it > lr_decay_iters:
