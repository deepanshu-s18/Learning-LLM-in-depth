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
