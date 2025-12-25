"""
Visualization utilities for reasoning benchmarks and scaling comparison plots.
"""

import os
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt


def plot_model_comparison(
    results: List[Tuple[str, float]],
    model_sizes: Optional[Dict[str, str]] = None,
    dataset_name: str = "GSM8K",
    save_path: str = "results/model_reasoning_comparison.png",
    show: bool = False
) -> None:
    """
    Plot bar chart comparing model accuracy across model sizes.
    
    Args:
        results: List of (model_name, accuracy_score)
        model_sizes: Mapping of model_name to size string (e.g. '80M', '7B')
        dataset_name: Name of benchmark dataset
        save_path: Filepath to save output plot
        show: Whether to display interactive plot
    """
    if not results:
        print("No results to plot.")
        return

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    labels = []
    scores = []

    for name, score in results:
        size_str = model_sizes.get(name, "") if model_sizes else ""
        label = f"{name}\n({size_str})" if size_str else name
        labels.append(label)
        scores.append(score)

    x = range(len(labels))

    plt.figure(figsize=(11, 6), dpi=150)
    bars = plt.bar(x, scores, color="#2E7D32", alpha=0.85, edgecolor="#1B5E20", width=0.55)

    # Add score annotations above bars
    for bar, score in zip(bars, scores):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.015,
            f"{score * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.xticks(x, labels, rotation=25, ha="right", fontsize=10)
    plt.ylabel("Accuracy", fontsize=12, fontweight="bold")
    plt.title(f"Few-Shot Chain of Thought Reasoning vs Model Size ({dataset_name})", fontsize=14, fontweight="bold", pad=15)
    plt.ylim(0, 1.0)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot saved to {save_path}")

    if show:
        plt.show()
    plt.close()
