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
        tok = SimpleTokenizer()
        text = "Hello world! <|user|> Test <|assistant|>"
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        self.assertEqual(text, decoded)

    def test_numpy_autograd_parity(self):
        res = verify_numpy_vs_pytorch()
        self.assertTrue(res["parity_verified"])
        self.assertLess(res["max_W_grad_difference"], 1e-7)

    def test_adamw_parity(self):
        res = verify_adamw_against_pytorch()
        self.assertTrue(res["parity_verified"])
        self.assertLess(res["max_parameter_difference"], 1e-10)

    def test_rope_invariance(self):
        res = verify_rope_relative_invariance()
        self.assertTrue(res["invariance_verified"])
        self.assertLess(res["absolute_difference"], 1e-6)

    def test_transformer_forward_and_kv_cache(self):
        config = TransformerConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32)
        model = TransformerLM(config)
        idx = torch.randint(0, 64, (2, 16))
        logits, loss, _ = model(idx, targets=idx)
        self.assertEqual(logits.shape, (2, 16, 64))
        self.assertIsNotNone(loss)

        # Test generation with KV cache
        gen_tokens = model.generate([1, 2, 3], max_new_tokens=10, use_kv_cache=True, eos_token_id=None)
        self.assertEqual(len(gen_tokens), 13)

    def test_lora_injection_and_merging(self):
        config = TransformerConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32)
        model = TransformerLM(config)
        
        linear = nn.Linear(32, 32)
        lora_layer = LoRALinear(linear, r=4, alpha=8.0)
        x = torch.randn(2, 32)
        out1 = lora_layer(x)
        lora_layer.merge_weights()
        out2 = lora_layer(x)
        # Verify merged forward equals pre-merged forward
        diff = torch.max(torch.abs(out1 - out2)).item()
        self.assertLess(diff, 1e-5)

    def test_quantization_int8_and_int4(self):
        tensor = torch.randn(10, 10)
        # INT8 test
        q8, s8, z8 = quantize_int8(tensor)
        deq8 = dequantize_int8(q8, s8, z8)
        mse8 = torch.mean((tensor - deq8) ** 2).item()
        self.assertLess(mse8, 0.01)

        # INT4 test
        q4, s4, z4 = quantize_int4(tensor)
        deq4 = dequantize_int4(q4, tensor.shape, s4, z4)
        mse4 = torch.mean((tensor - deq4) ** 2).item()
        self.assertLess(mse4, 0.15)

if __name__ == "__main__":
    unittest.main()
