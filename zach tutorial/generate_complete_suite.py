#!/usr/bin/env python3
"""
Master Generator for all 14 Zach LLM Interactive Colab Notebooks + Index Notebook + README.md
Uses AST parsing and smart classification to guarantee clean execution.
"""

import ast
import json
import math
import os
import re
import textwrap

BASE_DIR = "/Users/shobhitagnihotri/Desktop/internship/zach tutorial"
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def make_nb(cells, title):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": title,
                "provenance": []
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md(content):
    if isinstance(content, list):
        lines = [l if l.endswith("\n") else l + "\n" for l in content]
    else:
        lines = [l + "\n" for l in content.splitlines()]
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code(content):
    if isinstance(content, list):
        lines = [l if l.endswith("\n") else l + "\n" for l in content]
    else:
        lines = [l + "\n" for l in content.splitlines()]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

def save_nb(nb, name):
    p1 = os.path.join(NOTEBOOKS_DIR, f"{name}.ipynb")
    p2 = os.path.join(BASE_DIR, f"{name}.ipynb")
    for p in [p1, p2]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f" Saved: {name}.ipynb")

def read_source(fname):
    p = os.path.join(BASE_DIR, fname)
    if not os.path.exists(p):
        p = os.path.join(BASE_DIR, "original_docs", fname)
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def parse_blocks(text):
    tokens = text.split("```")
    blocks = []
    for i, token in enumerate(tokens):
        if i % 2 == 0:
            c = token.strip()
            if c:
                blocks.append(("markdown", c))
        else:
            lines = token.splitlines()
            if lines:
                first = lines[0].strip()
                if first in ["python", "py", "diff", "text", "mermaid", "json", "bash", "sh", "none", "cpp", "c"]:
                    lang = first
                    c = "\n".join(lines[1:]).strip()
                else:
                    lang = "python"
                    c = token.strip()
            else:
                lang = "python"
                c = ""
            if c:
                blocks.append(("code", lang, c))
    return blocks

def process_and_add_blocks(cells, orig_filename):
    raw_md = read_source(orig_filename)
    blocks = parse_blocks(raw_md)

    for block_type, *rest in blocks:
        if block_type == "markdown":
            content = rest[0]
            cells.append(md(content))
        elif block_type == "code":
            lang, code_content = rest
            
            # Check if it is text/mermaid/diff/json explicitly
            if lang in ["mermaid", "diff", "text", "json", "bash", "sh", "yaml", "html", "css", "markdown", "md"]:
                cells.append(md(f"```{lang}\n{code_content}\n```"))
                continue
                
            raw = code_content.strip()
            non_python_starters = [
                "A 2D coordinate plane", "A 3D visualization", "Diagram:", "A diagram illustrating",
                "INPUT:", "OUTPUT:", "// ALGORITHM", "<|user|>", "<|assistant|>", "<|system|>",
                "What is the primary cause", "What's the capital", "The primary cause",
                "Original Floats:", "Logits shape", "First number (", "--- Prepared Batch",
                "Input:          Kernel:", "Input (2×2):", "Imagine a timeline", "tensor([[[[",
                "Angles (m * theta_i)", ">>> torch.", ">>> a = torch"
            ]
            
            is_non_python = any(raw.startswith(s) or (s in raw[:100]) for s in non_python_starters) or ("→" in raw) or ("×" in raw)
            if is_non_python:
                cells.append(md(f"```text\n{raw}\n```"))
                continue
                
            # Try parsing Python AST with dedent
            dedented = textwrap.dedent(code_content)
            filtered_lines = [l for l in dedented.splitlines() if not l.strip().startswith('!') and not l.strip().startswith('%')]
            test_code = "\n".join(filtered_lines)
            
            try:
                ast.parse(test_code)
                # Valid standalone python code cell!
                cells.append(code(dedented))
            except SyntaxError:
                # If it's a code snippet or method fragment, format as a markdown python block
                cells.append(md(f"```python\n{dedented}\n```"))

# ----------------------------------------------------------------------
# 1. 01_Neural_Networks_From_Scratch.ipynb
# ----------------------------------------------------------------------
def build_01_nn():
    cells = []
    cells.append(md("""# 01. Neural Networks From Scratch: How AI Learns

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/01_Neural_Networks_From_Scratch.ipynb)

> **Tutorial Overview**: Master how neural networks learn from first principles. We will implement gradient descent, partial derivatives, the chain rule, and backpropagation from scratch in pure Python/NumPy, and visualize how the decision boundary evolves over time.
> **Original Source**: `nn.md`

---"""))

    cells.append(code("""# Setup & Imports
!pip install -q matplotlib numpy torch

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
print(" Setup complete! Pure NumPy & Matplotlib ready.")
"""))

    process_and_add_blocks(cells, "nn.md")

    cells.append(md("""## **Interactive Playground: Complete NumPy Neural Network & Decision Boundary Visualizer**
Let's assemble all the concepts into a complete, 2-layer Neural Network and train it on a non-linear dataset (two concentric circles / moons) to watch gradient descent separate the classes in real-time!"""))

    cells.append(code("""# Complete Pure NumPy 2-Layer Neural Network
class SimpleNeuralNet:
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1, lr=0.1):
        self.lr = lr
        # Initialize weights with small random numbers
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros((1, output_dim))
        
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
    
    def sigmoid_deriv(self, a):
        return a * (1.0 - a)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, y_hat):
        m = X.shape[0]
        # Binary Cross-Entropy / MSE gradient
        dL_dz2 = (y_hat - y) * self.sigmoid_deriv(y_hat)
        dL_dW2 = np.dot(self.a1.T, dL_dz2) / m
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True) / m
        
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_da1 * self.sigmoid_deriv(self.a1)
        dL_dW1 = np.dot(X.T, dL_dz1) / m
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True) / m
        
        # Gradient Descent Step
        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1

# Generate synthetic non-linear dataset (XOR / Circle)
N = 200
X = np.random.randn(N, 2)
y = ((X[:, 0]**2 + X[:, 1]**2) < 1.0).astype(float).reshape(-1, 1)

# Train network
net = SimpleNeuralNet(input_dim=2, hidden_dim=8, output_dim=1, lr=0.5)
losses = []
for epoch in range(1000):
    y_pred = net.forward(X)
    loss = np.mean((y_pred - y)**2)
    losses.append(loss)
    net.backward(X, y, y_pred)

print(f"Final Loss after 1000 epochs: {losses[-1]:.4f}")

# Plot Loss curve and Decision Boundary
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(losses, color='darkorange', lw=2)
axes[0].set_title("Training Loss (MSE) Over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)

# Decision Boundary Grid
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = net.forward(grid).reshape(xx.shape)
axes[1].contourf(xx, yy, probs, levels=20, cmap='RdBu_r', alpha=0.8)
axes[1].scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdBu_r', edgecolors='k')
axes[1].set_title("Learned Non-Linear Decision Boundary")
plt.tight_layout()
plt.show()
"""))
    save_nb(make_nb(cells, "01_Neural_Networks_From_Scratch"), "01_Neural_Networks_From_Scratch")

# ----------------------------------------------------------------------
# 2. 02_PyTorch_Deep_Dive.ipynb
# ----------------------------------------------------------------------
def build_02_pytorch():
    cells = []
    cells.append(md("""# 02. PyTorch Deep Dive: From Tensors to Training Loop

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/02_PyTorch_Deep_Dive.ipynb)

> **Tutorial Overview**: Build a complete, deep understanding of PyTorch. Tensors, Autograd, `nn.Module`, loss functions, optimizers, and constructing the industrial-strength training & evaluation loop.
> **Original Source**: `pytorch.md`

---"""))

    cells.append(code("""# Setup & Imports
!pip install -q torch torchvision matplotlib numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Version: {torch.__version__} on {device}")
"""))

    process_and_add_blocks(cells, "pytorch.md")

    cells.append(md("""## **Interactive Playground: End-to-End PyTorch Training on Multi-Class Classification**"""))
    cells.append(code("""# Complete Multi-Layer Perceptron (MLP) Classifier
class MLPClassifier(nn.Module):
    def __init__(self, in_features=2, hidden_dim=32, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Generate 3-class spiral dataset
N_points = 100
centers = [(-1.5, -1.0), (1.5, -1.0), (0.0, 1.5)]
X_data, y_data = [], []
for label, (cx, cy) in enumerate(centers):
    pts = np.random.randn(N_points, 2) * 0.4 + np.array([cx, cy])
    X_data.append(pts)
    y_data.append(np.full(N_points, label))

X = torch.tensor(np.vstack(X_data), dtype=torch.float32).to(device)
y = torch.tensor(np.concatenate(y_data), dtype=torch.long).to(device)

model = MLPClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.03)

# Training loop
loss_history = []
for epoch in range(150):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

print(f"Training Complete! Final Cross-Entropy Loss: {loss_history[-1]:.4f}")

# Plot loss
plt.figure(figsize=(8, 4))
plt.plot(loss_history, color='royalblue', lw=2)
plt.title("Cross-Entropy Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.show()
"""))
    save_nb(make_nb(cells, "02_PyTorch_Deep_Dive"), "02_PyTorch_Deep_Dive")

# ----------------------------------------------------------------------
# 3. 03_Adam_Optimizer_Demystified.ipynb
# ----------------------------------------------------------------------
def build_03_adam():
    cells = []
    cells.append(md("""# 03. Adam Optimizer Demystified: From SGD to Adaptive Moments

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/03_Adam_Optimizer_Demystified.ipynb)

> **Tutorial Overview**: Understand why standard SGD struggles in ravines and saddle points, how Momentum solves oscillations, how RMSprop scales learning rates, and how Adam combines both with Bias Correction.
> **Original Source**: `adam.md`

---"""))

    cells.append(code("""# Setup & Imports
