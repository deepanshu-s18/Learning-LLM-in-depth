"""
Dataset loading, subset selection, and preprocessing for GSM8K and SVAMP benchmarks.
"""

import os
import re
import json
import subprocess
from typing import List, Dict, Any, Optional
from datasets import load_dataset, Dataset


def load_gsm8k(split: str = "test", limit: Optional[int] = 50) -> Dataset:
    """
    Load the GSM8K dataset (main split).
    
    Args:
        split: 'train' or 'test'
        limit: Optional maximum number of samples to select
    
    Returns:
        HuggingFace Dataset subset
    """
    print(f"Loading GSM8K ({split} split)...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if limit and limit < len(dataset):
        dataset = dataset.select(range(limit))
    print(f"Loaded {len(dataset)} GSM8K samples.")
    return dataset


def load_svamp(repo_dir: str = "./SVAMP", limit: Optional[int] = 50) -> Optional[Dataset]:
    """
    Load the SVAMP dataset from local clone or GitHub repository.
    
    Args:
        repo_dir: Local path to SVAMP repository
        limit: Optional maximum number of samples to select
    
    Returns:
        Dataset subset or None if unavailable
    """
