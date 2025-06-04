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
