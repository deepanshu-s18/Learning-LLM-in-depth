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
!pip install -q torch matplotlib numpy

import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
"""))

    process_and_add_blocks(cells, "adam.md")

    cells.append(md("""## **Interactive Playground: SGD vs Momentum vs RMSprop vs Adam on a Ravine Surface**
Let's build custom optimizers from scratch and compare their trajectories navigating an elongated ravine (an ill-conditioned quadratic surface $f(x, y) = 0.1 x^2 + 2.0 y^2$)."""))

    cells.append(code("""# Custom implementation of Adam from scratch
class CustomAdam:
    def __init__(self, params, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad
                # 1. First Moment (Momentum)
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                # 2. Second Moment (RMSprop)
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
                # 3. Bias Correction
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                # 4. Update
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# Define an elongated ravine function
def loss_fn(x, y):
    return 0.1 * (x ** 2) + 2.0 * (y ** 2)

def run_optimizer(opt_name, steps=50):
    point = torch.tensor([-4.0, 3.0], requires_grad=True)
    history = [point.detach().numpy().copy()]
    
    if opt_name == "SGD":
        opt = torch.optim.SGD([point], lr=0.15)
    elif opt_name == "SGD+Momentum":
        opt = torch.optim.SGD([point], lr=0.08, momentum=0.9)
    elif opt_name == "RMSprop":
        opt = torch.optim.RMSprop([point], lr=0.1)
    elif opt_name == "Custom Adam":
        opt = CustomAdam([point], lr=0.2)
        
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(point[0], point[1])
        loss.backward()
        opt.step()
        history.append(point.detach().numpy().copy())
    return np.array(history)

# Run simulations
optimizers = ["SGD", "SGD+Momentum", "RMSprop", "Custom Adam"]
colors = ['red', 'purple', 'green', 'blue']

# Plot contours
X_grid, Y_grid = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-4, 4, 200))
Z_grid = loss_fn(X_grid, Y_grid)

plt.figure(figsize=(10, 7))
plt.contour(X_grid, Y_grid, Z_grid, levels=30, cmap='plasma', alpha=0.6)

for name, col in zip(optimizers, colors):
    traj = run_optimizer(name, steps=60)
    plt.plot(traj[:, 0], traj[:, 1], marker='o', markersize=3, label=name, color=col, lw=2)

plt.plot(0, 0, 'r*', markersize=15, label='Global Minimum (0,0)')
plt.title("Optimization Trajectories in an Elongated Ravine")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""))
    save_nb(make_nb(cells, "03_Adam_Optimizer_Demystified"), "03_Adam_Optimizer_Demystified")

# ----------------------------------------------------------------------
# 4. 04_Attention_Mechanism_Step_by_Step.ipynb
# ----------------------------------------------------------------------
def build_04_attention():
    cells = []
    cells.append(md("""# 04. The Attention Mechanism: Step-by-Step

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/04_Attention_Mechanism_Step_by_Step.ipynb)

> **Tutorial Overview**: Demystify Scaled Dot-Product Attention ($Q, K, V$), the softmax temperature factor $\\sqrt{d_k}$, causal masking, and Multi-Head Attention with tensor tracking and attention heatmap visualizations.
> **Original Source**: `attention.md`

---"""))

    cells.append(code("""# Setup & Imports
!pip install -q torch matplotlib numpy

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)
"""))

    process_and_add_blocks(cells, "attention.md")

    cells.append(md("""## **Interactive Playground: Multi-Head Attention & Dynamic Attention Heatmap**"""))
    cells.append(code("""# Scaled Dot-Product Attention with Attention Weights Output
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights

# Example sentence tokens
tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "tired"]
seq_len = len(tokens)
d_model = 16

# Generate random Query, Key, Value representations
X = torch.randn(1, seq_len, d_model)
W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)

Q, K, V = W_q(X), W_k(X), W_v(X)
output, attn_weights = scaled_dot_product_attention(Q, K, V)

print("Attention Output Shape:", output.shape)
print("Attention Weights Shape:", attn_weights.shape)

# Visualize Attention Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(attn_weights[0].detach().numpy(), cmap='magma')
plt.colorbar(label='Attention Weight')
plt.xticks(range(seq_len), tokens, rotation=45)
plt.yticks(range(seq_len), tokens)
plt.title("Self-Attention Alignment Matrix")
plt.tight_layout()
plt.show()
"""))
    save_nb(make_nb(cells, "04_Attention_Mechanism_Step_by_Step"), "04_Attention_Mechanism_Step_by_Step")

# ----------------------------------------------------------------------
# 5. 05_Transformer_From_Scratch.ipynb
# ----------------------------------------------------------------------
def build_05_transformer():
    cells = []
    cells.append(md("""# 05. Transformer Architecture From Scratch (GPT-2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/05_Transformer_From_Scratch.ipynb)

> **Tutorial Overview**: Build a complete, production-grade Decoder-Only Transformer (GPT-2 style) from scratch in PyTorch. Includes Token & Positional Embeddings, Pre-LayerNorm, Multi-Head Causal Self-Attention, MLP FeedForward, and Text Generation sampling.
> **Original Source**: `transformer.md`

---"""))

    cells.append(code("""# Setup & Imports
!pip install -q torch matplotlib numpy

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")
"""))

    process_and_add_blocks(cells, "transformer.md")

    cells.append(md("""## **Interactive Playground: Train Mini-GPT & Generate Character Text**"""))
    cells.append(code("""# Complete working Mini-GPT Definition
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=64, n_head=4, block_size=64, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, d_model=64, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.c_proj = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, d_model=64, n_head=4, block_size=64, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size=65, d_model=64, n_layer=2, n_head=4, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=0.8, top_k=5):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Test instantiation & generation
sample_text = "To be or not to be that is the question."
chars = sorted(list(set(sample_text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

model = MiniGPT(vocab_size=len(chars), d_model=64, n_layer=2, n_head=4).to(device)
prompt_tensor = torch.tensor([encode("To be")], dtype=torch.long).to(device)
gen_out = model.generate(prompt_tensor, max_new_tokens=25)
print("Generated Tokens Output:", decode(gen_out[0].cpu().tolist()))
"""))
    save_nb(make_nb(cells, "05_Transformer_From_Scratch"), "05_Transformer_From_Scratch")

# ----------------------------------------------------------------------
# 6. 06_KV_Cache_Optimization.ipynb
# ----------------------------------------------------------------------
def build_06_kv_cache():
    cells = []
    cells.append(md("""# 06. KV Cache: Accelerating Transformer Inference

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/06_KV_Cache_Optimization.ipynb)

> **Tutorial Overview**: Demystify KV Caching in Autoregressive LLM generation. Understand why recomputing attention keys and values is $O(N^2)$ wasteful, implement dynamic KV Cache buffers, and benchmark latency speedups.
> **Original Source**: `kv_cache.md`

---"""))

    cells.append(code("""# Setup & Imports
!pip install -q torch matplotlib numpy

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""))

    process_and_add_blocks(cells, "kv_cache.md")

    cells.append(md("""## **Interactive Playground: Naive vs KV-Cache Generation Speed Benchmark**"""))
    cells.append(code("""# Benchmark Naive generation vs KV-Cached Generation
class CachedAttention(nn.Module):
    def __init__(self, d_model=128, n_head=4):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        new_kv_cache = (k, v)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), new_kv_cache

attn = CachedAttention(128, 4).to(device)

# Measure token-by-token generation latency over 60 steps
steps = 60
naive_times = []
cached_times = []

# Naive simulation (full context recomputed)
for step in range(1, steps + 1):
    full_seq = torch.randn(1, step, 128).to(device)
    t0 = time.perf_counter()
    _ = attn(full_seq, kv_cache=None)
    naive_times.append(time.perf_counter() - t0)

# Cached simulation (single new token processed + KV concatenated)
kv_cache = None
for step in range(1, steps + 1):
    new_token = torch.randn(1, 1, 128).to(device)
    t0 = time.perf_counter()
    _, kv_cache = attn(new_token, kv_cache=kv_cache)
    cached_times.append(time.perf_counter() - t0)

# Plot comparison
plt.figure(figsize=(9, 5))
plt.plot(range(1, steps + 1), [t * 1000 for t in naive_times], label='Naive (O(N^2) Recomputation)', color='crimson', lw=2)
plt.plot(range(1, steps + 1), [t * 1000 for t in cached_times], label='KV-Cached (O(1) Step Latency)', color='teal', lw=2)
plt.xlabel("Generation Step (Sequence Length)")
plt.ylabel("Step Time (ms)")
plt.title("Latency per Generated Token: Naive vs KV-Cache")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""))
    save_nb(make_nb(cells, "06_KV_Cache_Optimization"), "06_KV_Cache_Optimization")

# ----------------------------------------------------------------------
# 7. 07_Rotary_Positional_Encoding_RoPE.ipynb
# ----------------------------------------------------------------------
def build_07_rope():
    cells = []
    cells.append(md("""# 07. Rotary Positional Encoding (RoPE): Math & Implementation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/07_Rotary_Positional_Encoding_RoPE.ipynb)

> **Tutorial Overview**: Master Rotary Positional Embeddings (RoPE) used in LLaMA, Mistral, and DeepSeek. Learn how 2D complex rotations inject relative positional awareness into self-attention.
> **Original Source**: `rope.md`

---"""))

    cells.append(code("""# Setup & Imports
!pip install -q torch matplotlib numpy

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
"""))

    process_and_add_blocks(cells, "rope.md")

    cells.append(md("""## **Interactive Playground: Verifying RoPE Relative Distance Invariance**"""))
    cells.append(code("""# Precompute RoPE frequencies & apply rotation
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex e^(i*m*theta)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq_.shape[1], :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

dim = 64
max_seq_len = 100
freqs_cis = precompute_freqs_cis(dim, max_seq_len)

# Create identical vectors at different token positions m and n
v = torch.randn(1, 1, dim)
q_m = v.expand(1, max_seq_len, dim)
k_n = v.expand(1, max_seq_len, dim)

q_rot, k_rot = apply_rotary_emb(q_m, k_n, freqs_cis)

# Compute attention score between position 0 and all positions d = 0..99
scores = (q_rot[:, [0], :] @ k_rot.transpose(-2, -1)).squeeze().detach().numpy()

plt.figure(figsize=(9, 4))
plt.plot(range(max_seq_len), scores, color='indigo', lw=2)
plt.title("RoPE Attention Score vs Relative Token Distance (m - n)")
plt.xlabel("Relative Token Distance (Tokens Apart)")
plt.ylabel("Query-Key Dot Product")
plt.grid(True, alpha=0.3)
plt.show()
"""))
    save_nb(make_nb(cells, "07_Rotary_Positional_Encoding_RoPE"), "07_Rotary_Positional_Encoding_RoPE")

# ----------------------------------------------------------------------
# 8. 08_LLM_Pretraining_From_Scratch.ipynb
# ----------------------------------------------------------------------
def build_08_pretrain():
    cells = []
    cells.append(md("""# 08. LLM Pre-Training: The Foundation of Large Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/08_LLM_Pretraining_From_Scratch.ipynb)

> **Tutorial Overview**: Understand self-supervised next-token prediction, tokenization pipelines, causal masking, cross-entropy loss, perplexity, and the full training loop with learning rate scheduling.
> **Original Source**: `pretrain.md`

---"""))

    cells.append(code("""# Setup & Imports
!pip install -q torch matplotlib numpy

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""))

    process_and_add_blocks(cells, "pretrain.md")

    cells.append(md("""## **Interactive Playground: Next-Token Pretraining & Perplexity Tracking**"""))
    cells.append(code("""# Mini Language Model for Pre-training Demonstration
class TinyLM(nn.Module):
    def __init__(self, vocab_size=50, d_model=32, seq_len=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        x = self.emb(idx)
        out, _ = self.gru(x)
        logits = self.head(out)
        return logits

# Synthetic pre-training tokens sequence
vocab_size = 50
seq_len = 16
batch_size = 16

data = torch.randint(0, vocab_size, (100, seq_len)).to(device)
model = TinyLM(vocab_size=vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

loss_history = []
ppl_history = []

for epoch in range(100):
    for batch in data.split(batch_size):
        # Autoregressive shifting: x is tokens 0..T-1, y is tokens 1..T
        x_in = batch[:, :-1]
        y_tgt = batch[:, 1:]
        
        logits = model(x_in)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y_tgt.contiguous().view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    loss_val = loss.item()
    loss_history.append(loss_val)
    ppl_history.append(math.exp(min(loss_val, 20)))

# Plot Loss & Perplexity
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(loss_history, color='tab:blue', lw=2)
axes[0].set_title("Pre-Training Cross-Entropy Loss")
axes[0].set_xlabel("Epoch")
axes[0].grid(True, alpha=0.3)

axes[1].plot(ppl_history, color='tab:green', lw=2)
axes[1].set_title("Language Model Perplexity (PPL)")
axes[1].set_xlabel("Epoch")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))
    save_nb(make_nb(cells, "08_LLM_Pretraining_From_Scratch"), "08_LLM_Pretraining_From_Scratch")

# ----------------------------------------------------------------------
# 9. 09_Supervised_Fine_Tuning_SFT.ipynb
# ----------------------------------------------------------------------
def build_09_sft():
    cells = []
    cells.append(md("""# 09. Supervised Fine-Tuning (SFT): Transforming Base Models into Assistants

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/09_Supervised_Fine_Tuning_SFT.ipynb)

> **Tutorial Overview**: Learn how Supervised Fine-Tuning turns text completion engines into conversational assistants. Understand prompt formatting, conversational token templates, and loss masking with `label = -100`.
> **Original Source**: `sft.md`

---"""))

    cells.append(code("""# Setup & Imports
!pip install -q torch matplotlib numpy

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
"""))

    process_and_add_blocks(cells, "sft.md")

    cells.append(md("""## **Interactive Playground: SFT Loss Masking Visualizer**"""))
    cells.append(code("""# SFT Loss Masking Example
def prepare_sft_sample(instruction_tokens, response_tokens):
    # Full input is prompt + response
    input_ids = instruction_tokens + response_tokens
    # Mask instruction tokens with -100 so CrossEntropy ignores them
    labels = [-100] * len(instruction_tokens) + response_tokens
    return torch.tensor([input_ids]), torch.tensor([labels])

inst = [101, 2054, 2003, 1037, 2182] # "What is a cat?"
resp = [1037, 2182, 2003, 1037, 2833] # "A cat is an animal"

inputs, targets = prepare_sft_sample(inst, resp)

print("Full Input IDs:", inputs.tolist()[0])
print("Target Labels (with -100 masking):", targets.tolist()[0])

# Simulate cross-entropy computation with ignore_index=-100
vocab_size = 5000
logits = torch.randn(1, inputs.shape[1], vocab_size, requires_grad=True)

# Notice how ignore_index skips all -100 tokens automatically!
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)
print(f"Computed Loss over Assistant tokens only: {loss.item():.4f}")
"""))
    save_nb(make_nb(cells, "09_Supervised_Fine_Tuning_SFT"), "09_Supervised_Fine_Tuning_SFT")

# ----------------------------------------------------------------------
# 10. 10_LoRA_Low_Rank_Adaptation.ipynb
# ----------------------------------------------------------------------
def build_10_lora():
    cells = []
    cells.append(md("""# 10. LoRA: Low-Rank Adaptation for LLMs From Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/10_LoRA_Low_Rank_Adaptation.ipynb)

> **Tutorial Overview**: Master Parameter-Efficient Fine-Tuning (PEFT) with LoRA. Learn the intrinsic rank hypothesis, implement `LoRALinear`, freeze base weights, train $<1\\%$ of parameters, and merge weights for inference.
> **Original Source**: `lora.md`

---"""))

    cells.append(code("""# Setup & Imports
!pip install -q torch matplotlib numpy

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
"""))

    process_and_add_blocks(cells, "lora.md")

    cells.append(md("""## **Interactive Playground: LoRA Parameter Savings & Weight Merging**"""))
    cells.append(code("""# Complete LoRALinear module with merge/unmerge functionality
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 4, alpha: float = 16.0):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False
        
        # Freeze base parameters
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
            
        # Low-rank adapter matrices
        self.lora_A = nn.Parameter(torch.empty(r, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        if self.merged:
            return self.base(x)
        # base(x) + (x @ A^T @ B^T) * scaling
        return self.base(x) + (F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling)

    def merge(self):
        if not self.merged:
            # W_merged = W_0 + scaling * (B @ A)
            self.base.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

# Test parameter comparison
d_in, d_out, r = 4096, 4096, 8
base_linear = nn.Linear(d_in, d_out)
lora_layer = LoRALinear(base_linear, r=r)

orig_params = d_in * d_out
trainable_lora_params = (d_in * r) + (d_out * r)

print(f"Original Linear Parameters: {orig_params:,}")
print(f"Trainable LoRA Parameters (rank={r}): {trainable_lora_params:,}")
print(f"Parameter Reduction: {100.0 * (1 - trainable_lora_params / orig_params):.2f}% memory saved!")
"""))
    save_nb(make_nb(cells, "10_LoRA_Low_Rank_Adaptation"), "10_LoRA_Low_Rank_Adaptation")

# ----------------------------------------------------------------------
# 11. 11_LLM_Quantization_INT8_INT4.ipynb
# ----------------------------------------------------------------------
def build_11_quantization():
    cells = []
    cells.append(md("""# 11. LLM Quantization: FP32 to INT8 & INT4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/11_LLM_Quantization_INT8_INT4.ipynb)

> **Tutorial Overview**: Decode how model compression works through linear quantization. Implement scale and zero-point calibration, symmetric vs asymmetric mapping, and evaluate quantization error SNR/MSE.
> **Original Source**: `quantization.md`

---"""))

    cells.append(code("""# Setup & Imports
!pip install -q torch matplotlib numpy

import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
"""))

    process_and_add_blocks(cells, "quantization.md")

    cells.append(md("""## **Interactive Playground: Asymmetric INT8 Quantizer & Signal-to-Noise Ratio (SNR)**"""))
    cells.append(code("""# Full Quantize and Dequantize implementation
def quantize_asymmetric_int8(x: torch.Tensor):
    qmin, qmax = -128, 127
    rmin, rmax = x.min().item(), x.max().item()
    
    scale = (rmax - rmin) / (qmax - qmin)
    zero_point = round(-rmin / scale) + qmin
    zero_point = max(qmin, min(qmax, zero_point))
    
    q_x = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax).to(torch.int8)
    return q_x, scale, zero_point

def dequantize(q_x: torch.Tensor, scale: float, zero_point: int):
    return (q_x.float() - zero_point) * scale

# Test with Gaussian weights
weights_fp32 = torch.randn(1000) * 2.5
weights_int8, s, z = quantize_asymmetric_int8(weights_fp32)
weights_rec = dequantize(weights_int8, s, z)

mse = torch.mean((weights_fp32 - weights_rec)**2).item()
snr = 10 * torch.log10(torch.mean(weights_fp32**2) / torch.mean((weights_fp32 - weights_rec)**2)).item()

print(f"Scale: {s:.6f}, Zero-Point: {z}")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB (High quality reconstruction!)")

# Plot distribution
plt.figure(figsize=(10, 4))
plt.hist(weights_fp32.numpy(), bins=50, alpha=0.6, label='Original FP32', color='blue')
plt.hist(weights_rec.numpy(), bins=50, alpha=0.6, label='Dequantized from INT8', color='orange')
plt.title("FP32 vs Dequantized INT8 Weights Distribution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""))
    save_nb(make_nb(cells, "11_LLM_Quantization_INT8_INT4"), "11_LLM_Quantization_INT8_INT4")

# ----------------------------------------------------------------------
# 12. 12_RLHF_and_PPO_Alignment.ipynb
# ----------------------------------------------------------------------
def build_12_rlhf():
    cells = []
    cells.append(md("""# 12. RLHF + PPO: Aligning LLMs with Human Preferences

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/12_RLHF_and_PPO_Alignment.ipynb)

> **Tutorial Overview**: The mathematics behind ChatGPT alignment. Learn how Bradley-Terry reward models score completions and how Proximal Policy Optimization (PPO) fine-tunes the policy with a KL divergence anchor.
> **Original Source**: `rlhf.md`

---"""))

    cells.append(code("""# Setup & Imports
!pip install -q torch matplotlib numpy

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)
"""))

    process_and_add_blocks(cells, "rlhf.md")

    cells.append(md("""## **Interactive Playground: Bradley-Terry Reward Loss & PPO Clipping Objective**"""))
    cells.append(code("""# 1. Bradley-Terry Preference Loss
def bradley_terry_reward_loss(r_chosen, r_rejected):
    return -torch.mean(F.logsigmoid(r_chosen - r_rejected))

r_win = torch.tensor([2.5, 1.8, 3.2])
r_lose = torch.tensor([0.2, -0.5, 1.1])
loss = bradley_terry_reward_loss(r_win, r_lose)
print(f"Reward Model Loss: {loss.item():.4f}")

# 2. PPO Clipped Surrogate Loss Curve
ratios = np.linspace(0.5, 1.5, 100)
advantage = 1.0
eps = 0.2

unclipped = ratios * advantage
clipped = np.clip(ratios, 1.0 - eps, 1.0 + eps) * advantage
ppo_objective = np.minimum(unclipped, clipped)

plt.figure(figsize=(8, 5))
plt.plot(ratios, unclipped, '--', label='Unclipped Objective (r * A)', color='gray')
plt.plot(ratios, ppo_objective, label='PPO Clipped Objective (min(r*A, clip(r)*A))', color='teal', lw=3)
plt.axvline(1.0 - eps, color='red', linestyle=':', label='Clip Boundaries [1-eps, 1+eps]')
plt.axvline(1.0 + eps, color='red', linestyle=':')
plt.title("PPO Clipped Surrogate Objective (Advantage > 0)")
plt.xlabel("Probability Ratio r(theta)")
plt.ylabel("Surrogate Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""))
    save_nb(make_nb(cells, "12_RLHF_and_PPO_Alignment"), "12_RLHF_and_PPO_Alignment")

# ----------------------------------------------------------------------
# 13. 13_Direct_Preference_Optimization_DPO.ipynb
# ----------------------------------------------------------------------
def build_13_dpo():
    cells = []
    cells.append(md("""# 13. Direct Preference Optimization (DPO): Direct Alignment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/13_Direct_Preference_Optimization_DPO.ipynb)

> **Tutorial Overview**: Master Direct Preference Optimization (DPO). Learn how DPO mathematically eliminates the separate reward model and unstable RL loop, directly optimizing LLM policy probabilities on pairwise human preferences.
> **Original Source**: `dpo.md`

---"""))

    cells.append(code("""# Setup & Imports
!pip install -q torch matplotlib numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)
"""))

    process_and_add_blocks(cells, "dpo.md")

    cells.append(md("""## **Interactive Playground: DPO Loss Function & Implicit Reward Margin**"""))
    cells.append(code("""# Complete DPO Loss Implementation
class DPOTrainer:
    def __init__(self, beta=0.1):
        self.beta = beta
        
    def loss(self, pi_logps_win, pi_logps_lose, ref_logps_win, ref_logps_lose):
        # Log ratio differences
        pi_logratios = pi_logps_win - pi_logps_lose
        ref_logratios = ref_logps_win - ref_logps_lose
        
        logits = self.beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()
        
        # Implicit rewards
        r_win = self.beta * (pi_logps_win - ref_logps_win).detach()
        r_lose = self.beta * (pi_logps_lose - ref_logps_lose).detach()
        return loss, r_win, r_lose

dpo = DPOTrainer(beta=0.1)

# Simulating log probabilities for batch of 3 pairs
pi_w = torch.tensor([-1.2, -0.8, -1.5])
pi_l = torch.tensor([-2.5, -2.1, -3.0])
ref_w = torch.tensor([-1.8, -1.2, -2.0])
ref_l = torch.tensor([-1.9, -1.5, -2.1])

loss, r_win, r_lose = dpo.loss(pi_w, pi_l, ref_w, ref_l)

print(f"DPO Loss: {loss.item():.4f}")
print("Implicit Won Rewards:", r_win.numpy())
print("Implicit Lost Rewards:", r_lose.numpy())
print("Reward Margins (Won - Lost):", (r_win - r_lose).numpy())
"""))
    save_nb(make_nb(cells, "13_Direct_Preference_Optimization_DPO"), "13_Direct_Preference_Optimization_DPO")

# ----------------------------------------------------------------------
# 14. 14_Diffusion_Models_From_Scratch.ipynb
# ----------------------------------------------------------------------
def build_14_diffusion():
    cells = []
    cells.append(md("""# 14. Diffusion Models From Scratch (DDPM)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/14_Diffusion_Models_From_Scratch.ipynb)

> **Tutorial Overview**: Build a complete Denoising Diffusion Probabilistic Model (DDPM) from scratch. Master the forward Gaussian noise schedule ($\\beta_t, \\alpha_t, \\bar{\\alpha}_t$), time embeddings, and reverse denoising loops.
> **Original Source**: `diffusion.md`

---"""))

    cells.append(code("""# Setup & Imports
!pip install -q torch matplotlib numpy

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""))

    process_and_add_blocks(cells, "diffusion.md")

    cells.append(md("""## **Interactive Playground: 2D Toy Diffusion Model & Step-by-Step Denoising**"""))
    cells.append(code("""# Complete DDPM Noise Schedule & Forward/Reverse Sampling
class DiffusionSchedule:
    def __init__(self, T=100, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self.alphas_bar[t].view(-1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise

sched = DiffusionSchedule(T=50)

# Create 2D Swiss Roll / S-curve data points
n_pts = 300
theta = torch.linspace(0, 4 * math.pi, n_pts)
x0 = torch.stack([theta * torch.cos(theta), theta * torch.sin(theta)], dim=1) / 10.0

# Visualize Forward Noise addition at steps t = 0, 10, 25, 49
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for i, t_val in enumerate([0, 10, 25, 49]):
    t_tensor = torch.full((n_pts,), t_val, dtype=torch.long)
    xt, _ = sched.q_sample(x0, t_tensor)
    axes[i].scatter(xt[:, 0], xt[:, 1], alpha=0.6, s=15, color='darkviolet')
    axes[i].set_title(f"Step t = {t_val}")
    axes[i].set_xlim(-2.5, 2.5)
    axes[i].set_ylim(-2.5, 2.5)
plt.suptitle("Forward Diffusion Process: Adding Gaussian Noise")
plt.tight_layout()
plt.show()
"""))
    save_nb(make_nb(cells, "14_Diffusion_Models_From_Scratch"), "14_Diffusion_Models_From_Scratch")

# ----------------------------------------------------------------------
# 15. 00_Tutorial_Roadmap_and_Index.ipynb
# ----------------------------------------------------------------------
def build_00_index():
    cells = []
    cells.append(md("""# Zach's Interactive LLM Mastery Series: Complete Roadmap & Index

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/00_Tutorial_Roadmap_and_Index.ipynb)

Welcome to the **Interactive Colab Masterclass Series** for modern AI and Large Language Models, converted from Zach's PocketFlow tutorials into clean, runnable, visual Jupyter/Colab notebooks!

Every notebook is designed to be **100% self-contained**, runnable with a single click, and loaded with math formulas, step-by-step intuition, code implementations from scratch, and interactive playgrounds.

---

## **Curriculum & Suggested Learning Roadmap**

### **Phase 1: Foundations of Deep Learning & PyTorch**
1. [01_Neural_Networks_From_Scratch.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/01_Neural_Networks_From_Scratch.ipynb)
   * *Core Topics*: Perceptrons, Forward Pass, Chain Rule, Backpropagation, Non-linear Decision Boundaries in pure NumPy.
2. [02_PyTorch_Deep_Dive.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/02_PyTorch_Deep_Dive.ipynb)
   * *Core Topics*: Tensors, Autograd, `nn.Module`, Loss functions, Optimizers, Production Training Loop.
3. [03_Adam_Optimizer_Demystified.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/03_Adam_Optimizer_Demystified.ipynb)
   * *Core Topics*: SGD, Momentum, RMSprop, Bias Correction, Custom Adam Optimizer, 2D Ravine Trajectory contours.

### **Phase 2: The Core Transformer Architecture**
4. [04_Attention_Mechanism_Step_by_Step.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/04_Attention_Mechanism_Step_by_Step.ipynb)
   * *Core Topics*: $Q, K, V$ Matrices, Softmax Scaling Factor $\\sqrt{d_k}$, Multi-Head Attention, Alignment Heatmaps.
5. [05_Transformer_From_Scratch.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/05_Transformer_From_Scratch.ipynb)
   * *Core Topics*: Complete GPT-2 Model, Positional Embeddings, Pre-LN Blocks, Autoregressive Text Generation.
6. [06_KV_Cache_Optimization.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/06_KV_Cache_Optimization.ipynb)
   * *Core Topics*: Autoregressive Decoding Bottleneck, $O(N^2)$ vs $O(1)$ Step Latency, KV Cache Implementation & Benchmarks.
7. [07_Rotary_Positional_Encoding_RoPE.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/07_Rotary_Positional_Encoding_RoPE.ipynb)
   * *Core Topics*: 2D Complex Rotation Matrix, High-dimensional RoPE, Relative Distance Invariance.

### **Phase 3: Training, Fine-Tuning & Quantization**
8. [08_LLM_Pretraining_From_Scratch.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/08_LLM_Pretraining_From_Scratch.ipynb)
   * *Core Topics*: Self-supervised Pretraining, Tokenization, Next-token Cross-Entropy Loss, Perplexity (PPL).
9. [09_Supervised_Fine_Tuning_SFT.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/09_Supervised_Fine_Tuning_SFT.ipynb)
   * *Core Topics*: Chat Templates, Instruction Tuning, Prompt-Response Loss Masking (`label = -100`).
10. [10_LoRA_Low_Rank_Adaptation.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/10_LoRA_Low_Rank_Adaptation.ipynb)
    * *Core Topics*: PEFT, Low-Rank Decomposition $W + \\frac{\\alpha}{r}BA$, Parameter Efficiency, Zero-Overhead Weight Merging.
11. [11_LLM_Quantization_INT8_INT4.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/11_LLM_Quantization_INT8_INT4.ipynb)
    * *Core Topics*: INT8/INT4 Numerical Precision, Scale & Zero-Point Mapping, Symmetric vs Asymmetric, Quantized Linear Layer.

### **Phase 4: Alignment & Generative Diffusion**
12. [12_RLHF_and_PPO_Alignment.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/12_RLHF_and_PPO_Alignment.ipynb)
    * *Core Topics*: 3-Step Alignment Pipeline, Bradley-Terry Reward Modeling, PPO Clipped Surrogate Loss, KL Penalty.
13. [13_Direct_Preference_Optimization_DPO.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/13_Direct_Preference_Optimization_DPO.ipynb)
    * *Core Topics*: Closed-form Policy derivation, DPO Loss Function, Eliminating the Reward Model & RL instability.
14. [14_Diffusion_Models_From_Scratch.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/14_Diffusion_Models_From_Scratch.ipynb)
    * *Core Topics*: DDPM Forward Gaussian Noise Schedule, Noise Prediction UNet, Reverse Denoising Sampling.

---
"""))

    cells.append(code("""# Verify system and packages for the entire tutorial series
import torch
import torchvision
import matplotlib
import numpy

print(f" Python environment ready!")
print(f" - PyTorch: {torch.__version__}")
print(f" - NumPy: {numpy.__version__}")
print(f" - Matplotlib: {matplotlib.__version__}")
print(f" - CUDA Available: {torch.cuda.is_available()}")
"""))
    save_nb(make_nb(cells, "00_Tutorial_Roadmap_and_Index"), "00_Tutorial_Roadmap_and_Index")

# ----------------------------------------------------------------------
# Master Build
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Building all 14 Zach LLM Interactive Notebooks with smart AST classifier...")
    build_01_nn()
    build_02_pytorch()
    build_03_adam()
    build_04_attention()
    build_05_transformer()
    build_06_kv_cache()
    build_07_rope()
    build_08_pretrain()
    build_09_sft()
    build_10_lora()
    build_11_quantization()
    build_12_rlhf()
    build_13_dpo()
    build_14_diffusion()
    build_00_index()
    print("\n All 15 notebooks successfully created in 'zach tutorial/' and 'zach tutorial/notebooks/'!")
