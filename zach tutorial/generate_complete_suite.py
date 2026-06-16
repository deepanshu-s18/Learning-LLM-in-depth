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
