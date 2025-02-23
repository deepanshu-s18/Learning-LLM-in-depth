"""
interactive_chat.py
Interactive CLI Playground for LLM-Lite.
Allows live prompt testing across all trained model checkpoints (Base, SFT, PPO, DPO),
with toggleable KV-Cache acceleration and real-time generation metrics.
"""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import SimpleTokenizer
from src.model.transformer import TransformerLM, TransformerConfig
from src.foundations.pytorch_engine import get_device, load_checkpoint
from src.tuning.lora import inject_lora

def run_chat_studio():
    device = get_device()
    tokenizer = SimpleTokenizer()
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=96,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        use_rope=True
    )

    models_dir = "models"
    available_checkpoints = {
        "1": ("Base Model", os.path.join(models_dir, "base_model.pt")),
        "2": ("SFT + LoRA Model", os.path.join(models_dir, "sft_lora_model.pt")),
        "3": ("PPO Aligned Model", os.path.join(models_dir, "ppo_model.pt")),
        "4": ("DPO Aligned Model", os.path.join(models_dir, "dpo_model.pt"))
    }

    print("\n" + "=" * 60)
    print("  💬 LLM-Lite Interactive Terminal Studio")
    print("=" * 60)
    print("Available Models:")
    for k, (name, path) in available_checkpoints.items():
        status = "Available" if os.path.exists(path) else "Not Found (Run main.py first)"
        print(f"  [{k}] {name} - ({status})")

    choice = input("\nSelect Model [1-4] (default: 2): ").strip() or "2"
    model_name, ckpt_path = available_checkpoints.get(choice, available_checkpoints["2"])

    if not os.path.exists(ckpt_path):
        print(f"\n[!] Checkpoint {ckpt_path} does not exist yet. Initializing fresh model...")
        model = TransformerLM(config).to(device)
    else:
        print(f"\nLoading {model_name} from {ckpt_path}...")
        model = TransformerLM(config).to(device)
        if "sft" in ckpt_path:
            inject_lora(model, r=4, alpha=8.0)
        load_checkpoint(model, ckpt_path, device=device)

    model.eval()
    use_kv = True
    print("\n" + "-" * 60)
