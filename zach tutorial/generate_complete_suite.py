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
