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
    print(f"Device:          {device}", flush=True)
    print(f"Reasoning Model: {args.reasoning_model}", flush=True)
    print(f"Reward Model:    {args.reward_model}", flush=True)
    print(f"Prompt:          {args.prompt}", flush=True)
    print(f"Parameters:      N={args.beams}, M={args.beam_width}, steps={args.max_steps}", flush=True)
    print("=" * 65, flush=True)

    # 1. Load Reasoning Model
    print(f"\n⏳ Loading reasoning model ({args.reasoning_model})...", flush=True)
    reasoning_tokenizer = AutoTokenizer.from_pretrained(args.reasoning_model, use_fast=True)
    if reasoning_tokenizer.pad_token_id is None and reasoning_tokenizer.eos_token_id is not None:
        reasoning_tokenizer.pad_token = reasoning_tokenizer.eos_token

    is_seq2seq = "t5" in args.reasoning_model.lower() or "bart" in args.reasoning_model.lower()

    if is_seq2seq:
        reasoning_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.reasoning_model,
            torch_dtype=torch.float32,
        ).to(device)
    else:
        dtype = torch.float16 if device.type in ["cuda", "mps"] else torch.float32
        reasoning_model = AutoModelForCausalLM.from_pretrained(
            args.reasoning_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
    reasoning_model.eval()

    # 2. Load Reward Model
    print(f"⏳ Loading reward model ({args.reward_model})...", flush=True)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model,
        torch_dtype=torch.float32,
    ).to(device)
    reward_model.eval()

    # 3. Run Beam Search
    print("\n🔍 Executing PRM-guided tree expansion...", flush=True)
    beams, graph = beam_search_with_prm(
        prompt=args.prompt,
        reasoning_model=reasoning_model,
        reasoning_tokenizer=reasoning_tokenizer,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        is_seq2seq=is_seq2seq,
        N=args.beams,
        M=args.beam_width,
        max_steps=args.max_steps,
        device=device,
    )

    # 4. Display Ranked Candidates
    print("\n" + "=" * 65, flush=True)
    print("🏆 Top Reasoning Trajectories (Ranked by PRM Score)", flush=True)
    print("=" * 65, flush=True)
    for rank, (text, score, node_id) in enumerate(beams[:3], 1):
        print(f"\n[Rank {rank}] (Node: {node_id} | PRM Score: {score:.4f})", flush=True)
        print("-" * 50, flush=True)
        cleaned = text.replace("<|system|>", "").replace("<|user|>", "").replace("<|assistant|>", "").strip()
        print(cleaned, flush=True)

    # 5. Plot and Save Search Tree
    os.makedirs(os.path.dirname(os.path.abspath(args.output_plot)), exist_ok=True)
    plot_trace_graph_tree_clean(graph, title="PRM-Guided Beam Search Tree", save_path=args.output_plot)
    print(f"\n✅ Search tree graph generated and saved to: {args.output_plot}", flush=True)


if __name__ == "__main__":
    main()
