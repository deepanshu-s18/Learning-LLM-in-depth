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
    for _ in range(runs):
        start = time.perf_counter()
        _ = model.generate(prompt_ids, max_new_tokens=gen_len, use_kv_cache=False)
        naive_times.append(time.perf_counter() - start)
    avg_naive_time = sum(naive_times) / runs
    naive_tok_sec = gen_len / max(avg_naive_time, 1e-6)

    # 2. Benchmark KV-Cache Generation (O(1) per step)
    cache_times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model.generate(prompt_ids, max_new_tokens=gen_len, use_kv_cache=True)
        cache_times.append(time.perf_counter() - start)
