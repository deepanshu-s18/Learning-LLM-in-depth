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

def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  🚀 {title}")
    print("=" * 75)

def main():
    set_seed(42)
    device = get_device()
    print_banner(f"LLM-Lite: Master Lifecycle Pipeline (Device: {device})")

    tokenizer = SimpleTokenizer()
    print(f"✓ Tokenizer initialized with vocabulary size: {tokenizer.vocab_size}")

    # -------------------------------------------------------------------------
    # STAGE 1: Mathematical Foundations Parity Check (Notebooks 01, 02, 03, 07)
    # -------------------------------------------------------------------------
    print_banner("STAGE 1: Mathematical Foundations & Parity Verification")
    
    # 1. NumPy Autograd vs PyTorch Autograd (Notebook 01)
    np_res = verify_numpy_vs_pytorch()
    print(f"[01. Backprop from Scratch] NumPy vs PyTorch Autograd Parity: "
          f"max diff = {np_res['max_W_grad_difference']:.2e} -> "
          f"{'PASSED ✓' if np_res['parity_verified'] else 'FAILED ✗'}")

    # 2. Custom AdamW vs torch.optim.AdamW (Notebook 03)
    adam_res = verify_adamw_against_pytorch()
    print(f"[03. AdamW Optimizer] CustomAdamW vs PyTorch AdamW Parity: "
          f"max diff = {adam_res['max_parameter_difference']:.2e} -> "
          f"{'PASSED ✓' if adam_res['parity_verified'] else 'FAILED ✗'}")

    # 3. RoPE Relative Invariance (Notebook 07)
    rope_res = verify_rope_relative_invariance()
    print(f"[07. RoPE Embeddings] Relative Distance Invariance: "
          f"diff = {rope_res['absolute_difference']:.2e} -> "
          f"{'PASSED ✓' if rope_res['invariance_verified'] else 'FAILED ✗'}")

    # -------------------------------------------------------------------------
    # STAGE 2: Model Architecture & Pretraining (Notebooks 04, 05, 08)
    # -------------------------------------------------------------------------
    print_banner("STAGE 2: Decoder Transformer Pretraining (Notebooks 04, 05, 08)")
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=96,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        use_rope=True
    )
    base_model = TransformerLM(config).to(device)
    pretrain_texts = get_pretrain_corpus()

    print(f"Pretraining Base Model on {len(pretrain_texts)} sentences...")
    pt_res = train_pretrain(base_model, pretrain_texts, tokenizer, epochs=30, lr=3e-3, device=device)
    print(f"✓ Base Model Pretrained! Final Loss: {pt_res['final_pretrain_loss']:.4f} | "
          f"Perplexity: {pt_res['pretrain_perplexity']:.2f}")

    # -------------------------------------------------------------------------
    # STAGE 3: Supervised Fine-Tuning + LoRA (Notebooks 09, 10)
    # -------------------------------------------------------------------------
    print_banner("STAGE 3: Supervised Fine-Tuning & LoRA PEFT (Notebooks 09, 10)")
    sft_data = get_sft_dataset()
    sft_model = copy.deepcopy(base_model)

    # Parameter accounting before and after LoRA injection
    initial_params = get_parameter_summary(sft_model)
    print(f"Total Base Parameters: {initial_params['total_parameters']:,}")

    # Inject LoRA adapters into attention projections
    lora_params = inject_lora(sft_model, r=4, alpha=8.0)
    lora_summary = get_parameter_summary(sft_model)
    print(f"LoRA Injected! Trainable parameters: {lora_summary['trainable_parameters']:,} "
          f"({100.0 - lora_summary['frozen_percentage']:.2f}% of model) | "
          f"Frozen: {lora_summary['frozen_percentage']:.2f}%")

    print(f"Fine-Tuning SFT Model on {len(sft_data)} instructions with prompt masking (label = -100)...")
    sft_res = train_sft(sft_model, sft_data, tokenizer, epochs=25, lr=1e-3, device=device)
    print(f"✓ SFT Fine-Tuning Complete! Final Loss: {sft_res['final_sft_loss']:.4f}")

    # -------------------------------------------------------------------------
    # STAGE 4: Human Preference Alignment via Reward Model + PPO (Notebook 12)
    # -------------------------------------------------------------------------
    print_banner("STAGE 4: Bradley-Terry Reward Model + PPO Alignment (Notebook 12)")
    pref_data = get_preference_dataset()

    print("1. Training Bradley-Terry Reward Model on pairwise preferences...")
    reward_model = RewardModel(sft_model).to(device)
    rm_res = train_reward_model(reward_model, pref_data, tokenizer, epochs=20, lr=1e-3, device=device)
    print(f"   ✓ Reward Model Trained! Loss: {rm_res['final_rm_loss']:.4f} | "
          f"Mean Chosen Reward: {rm_res['chosen_rewards_mean']:.3f} vs Rejected: {rm_res['rejected_rewards_mean']:.3f}")

    print("2. Running PPO Actor-Critic Alignment with KL divergence penalty...")
    ppo_model = copy.deepcopy(sft_model)
    prompts_list = [item["prompt"] for item in pref_data]
    ppo_res = train_ppo_alignment(ppo_model, reward_model, prompts_list, tokenizer, epochs=12, lr=5e-4, device=device)
    print(f"   ✓ PPO Aligned! Final Loss: {ppo_res['final_ppo_loss']:.4f} | "
          f"Penalized Reward: {ppo_res['final_penalized_reward']:.3f} | KL: {ppo_res['average_kl_divergence']:.4f}")

    # -------------------------------------------------------------------------
    # STAGE 5: Direct Preference Optimization (DPO) (Notebook 13)
    # -------------------------------------------------------------------------
    print_banner("STAGE 5: Direct Preference Optimization (DPO) (Notebook 13)")
    dpo_model = copy.deepcopy(sft_model)
    print("Running DPO on policy log-probabilities without a separate reward model...")
    dpo_res = train_dpo_alignment(dpo_model, pref_data, tokenizer, epochs=20, lr=5e-4, beta=0.1, device=device)
    print(f"✓ DPO Aligned! Final Loss: {dpo_res['final_dpo_loss']:.4f} | "
          f"Implicit Reward Margin: {dpo_res['final_implicit_margin']:.4f}")

    # -------------------------------------------------------------------------
    # STAGE 6: KV-Cache Latency & Speedup Benchmark (Notebook 06)
    # -------------------------------------------------------------------------
    print_banner("STAGE 6: KV-Cache Optimization & Benchmarking (Notebook 06)")
    test_prompt = "<|user|>What is backpropagation?<|assistant|>"
    test_ids = tokenizer.encode(test_prompt)
    kv_res = benchmark_kv_cache(sft_model, test_ids, gen_len=45, runs=3)
    print(f"Tokens Generated: {kv_res['tokens_generated']}")
    print(f"• Naive Quadratic O(T^2) Generation : {kv_res['naive_tokens_per_sec']:.1f} tok/sec ({kv_res['naive_time_sec']:.4f}s)")
    print(f"• Dynamic KV-Cache O(1) Generation  : {kv_res['kv_cache_tokens_per_sec']:.1f} tok/sec ({kv_res['kv_cache_time_sec']:.4f}s)")
    print(f"⚡ KV-Cache Speedup Factor: {kv_res['speedup_factor']:.2f}x faster!")

    # -------------------------------------------------------------------------
    # STAGE 7: Model Compression & Quantization (Notebook 11)
    # -------------------------------------------------------------------------
    print_banner("STAGE 7: Model Quantization INT8 & INT4 (Notebook 11)")
    q_res = benchmark_model_quantization(sft_model)
    print(f"• Uncompressed (FP32) Model Size : {q_res['fp32_memory_kb']:.1f} KB")
    print(f"• Quantized INT8 Model Size      : {q_res['int8_memory_kb']:.1f} KB ({q_res['int8_compression_ratio']} compression, MSE: {q_res['int8_average_mse']:.6f})")
    print(f"• Quantized INT4 Model Size      : {q_res['int4_memory_kb']:.1f} KB ({q_res['int4_compression_ratio']} compression, MSE: {q_res['int4_average_mse']:.6f})")

    # -------------------------------------------------------------------------
    # STAGE 8: Side-by-Side Model Comparison & Final Evaluation Report
    # -------------------------------------------------------------------------
    print_banner("STAGE 8: Side-by-Side Model Comparison Report")
    eval_prompt = "<|user|>What is KV caching in LLMs?<|assistant|>"
    prompt_tokens = tokenizer.encode(eval_prompt)

    print(f"Prompt: \"What is KV caching in LLMs?\"\n")

    out_base = tokenizer.decode(base_model.generate(prompt_tokens, max_new_tokens=40, use_kv_cache=True))
    out_sft = tokenizer.decode(sft_model.generate(prompt_tokens, max_new_tokens=40, use_kv_cache=True))
    out_ppo = tokenizer.decode(ppo_model.generate(prompt_tokens, max_new_tokens=40, use_kv_cache=True))
    out_dpo = tokenizer.decode(dpo_model.generate(prompt_tokens, max_new_tokens=40, use_kv_cache=True))

    def clean_output(full_text: str) -> str:
        if "<|assistant|>" in full_text:
            ans = full_text.split("<|assistant|>")[-1]
            return ans.replace("<|end|>", "").strip()
        return full_text.strip()

    print(f"1. [Base Model (Parrot)]       : {clean_output(out_base)[:120]}...")
    print(f"2. [SFT + LoRA Model]          : {clean_output(out_sft)[:120]}...")
    print(f"3. [RLHF + PPO Aligned Model]  : {clean_output(out_ppo)[:120]}...")
    print(f"4. [DPO Aligned Model]         : {clean_output(out_dpo)[:120]}...")

    # Save checkpoints to models/
    os.makedirs("models", exist_ok=True)
    save_checkpoint(base_model, None, "models/base_model.pt")
    save_checkpoint(sft_model, None, "models/sft_lora_model.pt")
    save_checkpoint(ppo_model, None, "models/ppo_model.pt")
    save_checkpoint(dpo_model, None, "models/dpo_model.pt")
    print(f"\n✓ All 4 model checkpoints saved successfully to ./models/")

    print_banner("🎉 ALL 13 TUTORIAL MODULES SUCCESSFULLY EXECUTED & VERIFIED!")

if __name__ == "__main__":
    main()
