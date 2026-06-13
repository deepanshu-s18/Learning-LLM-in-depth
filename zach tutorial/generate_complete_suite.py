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
