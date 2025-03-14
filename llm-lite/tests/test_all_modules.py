"""
tests/test_all_modules.py
Unit tests verifying every module and mathematical property across all 13 tutorial topics.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

from data.dataset import SimpleTokenizer, get_pretrain_corpus, get_sft_dataset, get_preference_dataset
from src.foundations.numpy_autograd import NumpyLinear, NumpySigmoid, NumpyMSELoss, verify_numpy_vs_pytorch
from src.optimization.custom_adamw import CustomAdamW, verify_adamw_against_pytorch
from src.model.rope import precompute_rope_frequencies, apply_rope, verify_rope_relative_invariance
from src.model.transformer import TransformerLM, TransformerConfig
from src.model.kv_cache import benchmark_kv_cache
from src.tuning.lora import LoRALinear, inject_lora, get_parameter_summary
from src.compression.quantizer import quantize_int8, dequantize_int8, quantize_int4, dequantize_int4

class TestLLMLite(unittest.TestCase):
    def test_tokenizer(self):
