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

def benchmark_kv_cache(
    model: TransformerLM,
    prompt_ids: List[int],
    gen_len: int = 40,
    runs: int = 3
) -> Dict[str, Any]:
    """
    Benchmarks generation latency and throughput comparing Naive vs KV-Cache decoding.
    """
    model.eval()

    # 1. Benchmark Naive Generation (O(T^2))
    naive_times = []
