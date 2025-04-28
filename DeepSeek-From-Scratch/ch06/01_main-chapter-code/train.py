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
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 1..0
    return min_lr + coeff * (learning_rate - min_lr)


# --- Main Training Script ---
if __name__ == '__main__':
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Metadata and Initialize Model
    meta_path = os.path.join(data_dir, 'meta.json')
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        vocab_size = meta['vocab_size']
    except FileNotFoundError:
        print(f"Error: meta.json not found in {data_dir}.")
        print("Please run prepare.py first to tokenize the dataset.")
        exit(1)

    # Model configuration for a ~18.5M parameter model
    # "FLAGSHIP" CONFIGURATION FOR RTX 4090 (~130M parameters)
    model_args = ModelArgs(
        d_model=768,                # Standard "small" model dimension
        n_layers=12,                # 12 layers deep
        num_heads=12,               # 12 attention heads (d_head=64)
        d_latent=192,               # Latent dim for MLA (3 * d_head)
        d_rope=64,                  # RoPE dim, same as d_head
        moe_n_routed_experts=16,    # A good number of experts to allow specialization
        moe_n_shared_experts=1,     # One generalist expert for common knowledge
        moe_top_k=4,                # Route to the best 4 experts per token
        moe_routed_hidden=512,      # Each expert is reasonably sized
        vocab_size=vocab_size,      # 50257 from our data
        max_seq_len=block_size
    )
    
    model = MiniDeepSeek(model_args)
    model.to(device)
    
    # --- Calculate and Print Total Parameters ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel initialized on {device}")
    print(f"  -> Total parameters: {total_params/1e6:.2f}M")
    print(f"  -> Trainable parameters: {trainable_params/1e6:.2f}M\n")

    # 2. Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))

    # 3. Training Loop
    t0 = time.time()
    best_val_loss = float('inf') # Initialize with infinity
    for iter_num in range(max_iters):
        # --- Update Learning Rate ---
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # --- Training Step ---
        model.train()
        X, Y = get_batch('train')
        
        # Forward pass with Automatic Mixed Precision
        with ctx:
            outputs = model(X, targets=Y)
            loss = outputs['loss']
        
        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # --- Logging ---
        if iter_num % log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")

        # --- Evaluation Step ---
        if iter_num % eval_interval == 0 and iter_num > 0:
            model.eval()
            losses = torch.zeros(eval_iters)
            print("Running evaluation...")
            with torch.no_grad():
                for k in range(eval_iters):
                    X, Y = get_batch('val')
                    with ctx:
                        outputs = model(X, targets=Y)
                        losses[k] = outputs['loss'].item()
            
            val_loss = losses.mean()
            print(f"iter {iter_num}: validation loss {val_loss:.4f}")

            # --- Checkpointing ---
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'val_loss': val_loss.item(),
            }

            # 1. Save the best model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = os.path.join(out_dir, 'best_ckpt.pt')
                print(f"New best validation loss: {val_loss:.4f}. Saving best checkpoint to {best_ckpt_path}")
                torch.save(checkpoint, best_ckpt_path)

            # 2. Save a periodic checkpoint every 500 iterations
            save_interval = 500
            if iter_num % save_interval == 0 and iter_num > 0:
                periodic_ckpt_path = os.path.join(out_dir, f'ckpt_iter_{iter_num}.pt')
                print(f"Saving periodic checkpoint to {periodic_ckpt_path}")
                torch.save(checkpoint, periodic_ckpt_path)

    print("\nTraining complete.")