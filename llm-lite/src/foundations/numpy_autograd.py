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
    
    x_np = np.random.randn(batch_size, in_dim).astype(np.float64)
    y_target_np = np.random.randn(batch_size, out_dim).astype(np.float64)
    
    # 1. Pure NumPy Forward & Backward
    np_layer = NumpyLinear(in_dim, out_dim, seed=42)
    np_act = NumpySigmoid()
    np_loss = NumpyMSELoss()
    
    z_np = np_layer.forward(x_np)
    a_np = np_act.forward(z_np)
    loss_val_np = np_loss.forward(a_np, y_target_np)
    
    d_loss = np_loss.backward(a_np, y_target_np)
    d_z = np_act.backward(d_loss)
    d_x = np_layer.backward(d_z)
    
    # 2. PyTorch Autograd with exact same weights
    x_pt = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)
    y_target_pt = torch.tensor(y_target_np, dtype=torch.float64)
    
    pt_linear = nn.Linear(in_dim, out_dim, bias=True).to(torch.float64)
    with torch.no_grad():
        pt_linear.weight.copy_(torch.tensor(np_layer.W.T))
        pt_linear.bias.copy_(torch.tensor(np_layer.b.squeeze(0)))
        
    z_pt = pt_linear(x_pt)
    a_pt = torch.sigmoid(z_pt)
    loss_pt = torch.mean((a_pt - y_target_pt) ** 2)
    loss_pt.backward()
    
    # Measure discrepancies
    w_grad_diff = float(np.max(np.abs(np_layer.dW - pt_linear.weight.grad.numpy().T)))
    b_grad_diff = float(np.max(np.abs(np_layer.db - pt_linear.bias.grad.numpy())))
    x_grad_diff = float(np.max(np.abs(d_x - x_pt.grad.numpy())))
    
    return {
        "numpy_loss": loss_val_np,
        "pytorch_loss": float(loss_pt.item()),
        "max_W_grad_difference": w_grad_diff,
        "max_b_grad_difference": b_grad_diff,
        "max_x_grad_difference": x_grad_diff,
        "parity_verified": (w_grad_diff < 1e-7 and b_grad_diff < 1e-7 and x_grad_diff < 1e-7)
    }
