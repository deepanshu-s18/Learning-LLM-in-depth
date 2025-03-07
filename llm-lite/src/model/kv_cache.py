"""
src/model/kv_cache.py
Notebook 06: KV Cache Optimization & Benchmarking.
Evaluates the latency, throughput (tokens/sec), and computational speedup of
KV Caching over naive quadratic O(T^2) recomputation.
"""

import time
import torch
from typing import Dict, Any, List
from src.model.transformer import TransformerLM

