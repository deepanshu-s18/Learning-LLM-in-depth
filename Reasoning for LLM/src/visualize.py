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
