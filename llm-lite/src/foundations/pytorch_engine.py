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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clip_gradients(parameters, max_norm: float = 1.0) -> float:
    """Clips parameter gradients by L2 norm to prevent exploding gradients."""
    return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm))

def save_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], filepath: str, extra_meta: Dict[str, Any] = None) -> None:
    """Serializes model weights, optimizer state, and training metadata to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metadata": extra_meta or {}
    }
    torch.save(state, filepath)

def load_checkpoint(model: nn.Module, filepath: str, optimizer: Optional[torch.optim.Optimizer] = None, device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
    """Deserializes model checkpoint from disk."""
    state = torch.load(filepath, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if optimizer and state.get("optimizer_state_dict"):
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state.get("metadata", {})
