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
        self.b = np.zeros((1, out_features))
        
        # Cache for backpropagation
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: z = x @ W + b"""
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass using the chain rule:
        dW = x.T @ grad_output
        db = sum(grad_output, axis=0, keepdims=True)
        dx = grad_output @ W.T
        """
        self.dW = np.dot(self.x.T, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.W.T)
