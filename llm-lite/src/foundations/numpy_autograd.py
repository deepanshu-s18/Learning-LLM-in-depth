"""
src/foundations/numpy_autograd.py
Notebook 01: Neural Networks & Backpropagation From Scratch in Pure NumPy.
Implements forward and backward passes using the multivariate chain rule and verifies parity with PyTorch Autograd.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict

class NumpyLinear:
    """
    Fully connected linear layer: y = x @ W + b.
    Computes analytical gradients dW, db, and dx during backpropagation.
    """
    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        np.random.seed(seed)
        # Xavier/Glorot initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
