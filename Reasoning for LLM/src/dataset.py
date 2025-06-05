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
    if not os.path.exists(repo_dir):
        print(f"SVAMP directory not found at {repo_dir}. Cloning from GitHub...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/arkilpatel/SVAMP.git", repo_dir],
                check=True,
                capture_output=True,
                text=True
            )
        except Exception as e:
            print(f"Failed to clone SVAMP: {e}")
            return None

    json_path = os.path.join(repo_dir, "SVAMP.json")
    if not os.path.exists(json_path):
        print(f"SVAMP.json not found at {json_path}")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Format into consistent keys
    formatted = []
    for item in raw_data:
        body = item.get("Body", "").strip()
        question = item.get("Question", "").strip()
        full_q = f"{body} {question}".strip()
        answer = str(item.get("Answer", "")).strip()
        formatted.append({
            "question": full_q,
            "answer": answer,
            "id": item.get("ID", "")
        })

    dataset = Dataset.from_list(formatted)
    if limit and limit < len(dataset):
        dataset = dataset.select(range(limit))
    print(f"Loaded {len(dataset)} SVAMP samples.")
    return dataset


def extract_ground_truth_answer(raw_answer: str) -> str:
    """
    Extract the numeric answer from ground truth strings.
    For GSM8K, extracts text following '####'.
    """
    if "####" in raw_answer:
        return raw_answer.split("####")[-1].strip().replace(",", "")
    return raw_answer.strip().replace(",", "")


def extract_predicted_answer(text: str) -> Optional[str]:
    """
    Extract the final numeric prediction from model generation.
    Handles integers, decimals, and negative numbers.
    """
    text = text.replace(",", "")
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if not numbers:
        return None
    return numbers[-1].strip()


def is_answer_match(pred: Optional[str], gt: str) -> bool:
    """Compare predicted numeric string with ground truth string."""
    if pred is None or not pred.strip():
        return False
    
    clean_pred = pred.strip().lstrip("0") or "0"
    clean_gt = gt.strip().lstrip("0") or "0"

    # Direct match
    if clean_pred == clean_gt:
        return True

    # Numeric float comparison
    try:
        if abs(float(pred) - float(gt)) < 1e-4:
            return True
    except ValueError:
        pass

    return False

