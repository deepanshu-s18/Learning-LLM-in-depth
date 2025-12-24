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
