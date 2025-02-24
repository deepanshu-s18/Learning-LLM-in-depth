"""
main.py
LLM-Lite: Unified Generative AI Engine From Scratch.
Integrates and runs every single algorithm from Notebooks 01 through 13 of the Zach Tutorial series:
1. NumPy Backpropagation & Chain Rule
2. PyTorch Engine & Checkpointing
3. Custom AdamW with Decoupled Decay
4. Scaled Dot-Product & Multi-Head Attention
5. Complete Decoder-Only Transformer Architecture
6. KV-Cache O(1) Decoding Engine
7. Rotary Positional Embeddings (RoPE)
8. Causal Next-Token Pretraining & Perplexity
9. Supervised Fine-Tuning with Prompt Loss Masking
10. Parameter-Efficient Fine-Tuning (LoRA)
11. INT8 and INT4 Quantization with Bit Packing
12. Bradley-Terry Reward Modeling + PPO RLHF (with KL penalty)
13. Direct Preference Optimization (DPO)
"""

import sys
import os
import copy
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import SimpleTokenizer, get_pretrain_corpus, get_sft_dataset, get_preference_dataset
from src.foundations.numpy_autograd import verify_numpy_vs_pytorch
from src.foundations.pytorch_engine import get_device, set_seed, save_checkpoint
from src.optimization.custom_adamw import verify_adamw_against_pytorch
from src.model.rope import verify_rope_relative_invariance
from src.model.transformer import TransformerLM, TransformerConfig
from src.model.kv_cache import benchmark_kv_cache
from src.tuning.pretrain import train_pretrain
from src.tuning.sft import train_sft
from src.tuning.lora import inject_lora, get_parameter_summary
from src.alignment.reward_model import RewardModel, train_reward_model
from src.alignment.ppo_aligner import train_ppo_alignment
from src.alignment.dpo_aligner import train_dpo_alignment
from src.compression.quantizer import benchmark_model_quantization
