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
    args = parser.parse_args()

    print("=" * 60)
    print("🧠 Chain of Thought (CoT) Reasoning Evaluation for LLMs")
    print(f"📊 Dataset: {args.dataset.upper()} | Samples: {args.limit}")
    print("=" * 60)

    # 1. Load Dataset
    if args.dataset.lower() == "gsm8k":
        dataset = load_gsm8k(split="test", limit=args.limit)
    else:
        dataset = load_svamp(limit=args.limit)

    if dataset is None or len(dataset) == 0:
        print("Error: Could not load dataset.")
        return

    # 2. Run Evaluations
    results = []

    for model_name in args.models:
        if model_name in SEQ2SEQ_MODELS:
            model_id = SEQ2SEQ_MODELS[model_name]
            evaluator = Seq2SeqEvaluator(model_id)
            if args.preview > 0:
                evaluator.preview_predictions(dataset, prefix=FEW_SHOT_COT_PREFIX, num_samples=args.preview)
            acc = evaluator.evaluate(dataset, prefix=FEW_SHOT_COT_PREFIX)
            results.append((model_name, acc))

        elif model_name in DECODER_MODELS:
            model_id = DECODER_MODELS[model_name]
            evaluator = DecoderEvaluator(model_id)
            if args.preview > 0:
                evaluator.preview_predictions(dataset, prefix=FEW_SHOT_COT_PREFIX, num_samples=args.preview)
            acc = evaluator.evaluate(dataset, prefix=FEW_SHOT_COT_PREFIX)
            results.append((model_name, acc))

        else:
            print(f"Unknown model '{model_name}'. Available: {list(SEQ2SEQ_MODELS.keys()) + list(DECODER_MODELS.keys())}")

    # 3. Print Summary
    print("\n" + "=" * 60)
    print("🏆 Benchmark Summary")
    print("=" * 60)
    for model_name, acc in results:
        size = MODEL_SIZES.get(model_name, "N/A")
        print(f"• {model_name:<20} ({size:<6}): {acc:.2%}")

    # 4. Save results to JSON
    os.makedirs("results", exist_ok=True)
    summary_data = {
        "dataset": args.dataset,
        "sample_limit": args.limit,
        "results": [{"model": m, "size": MODEL_SIZES.get(m, "N/A"), "accuracy": acc} for m, acc in results]
    }
    with open("results/evaluation_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # 5. Plot Comparison
    plot_model_comparison(results, model_sizes=MODEL_SIZES, dataset_name=args.dataset.upper(), save_path=args.output_plot)


if __name__ == "__main__":
    main()
