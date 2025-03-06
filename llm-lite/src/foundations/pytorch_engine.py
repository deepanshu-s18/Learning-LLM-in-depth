"""
src/foundations/pytorch_engine.py
Notebook 02: PyTorch Deep Dive & Production Training Harness.
Provides device abstraction (MPS/CUDA/CPU), autograd utilities, gradient clipping,
checkpoint serialization, and training tracking.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

def get_device() -> torch.device:
    """Detects and returns the best available hardware accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int = 42) -> None:
    """Enforces deterministic execution across random, numpy, and torch."""
    random.seed(seed)
