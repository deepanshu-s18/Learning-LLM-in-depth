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
        return grad_input


class NumpySigmoid:
    """
    Sigmoid activation: sigma(x) = 1 / (1 + exp(-x)).
    Derivative: d/dx sigma(x) = sigma(x) * (1 - sigma(x)).
    """
    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_clipped = np.clip(x, -50.0, 50.0)
        self.output = 1.0 / (1.0 + np.exp(-x_clipped))
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.output * (1.0 - self.output)


class NumpyMSELoss:
    """
    Mean Squared Error loss: L = (1 / N) * sum((y_pred - y_true)^2).
    Derivative: dL / dy_pred = (2 / N) * (y_pred - y_true).
    """
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n = y_pred.shape[0] * y_pred.shape[1]
        return (2.0 / n) * (y_pred - y_true)


def verify_numpy_vs_pytorch() -> Dict[str, float]:
    """
    Mathematically verifies that our pure NumPy backprop engine produces
    exact gradient parity with PyTorch Autograd.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    in_dim, out_dim = 4, 2
    batch_size = 3
