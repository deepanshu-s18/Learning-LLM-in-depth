<div align="center">

# llm-lite

**I built a complete language model stack from scratch — no HuggingFace, no shortcuts.**

Everything from raw bytes to a deployable chat model: tokenizer, transformer, optimizer, fine-tuning, alignment, and compression — implemented from scratch in Python and PyTorch.

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Why I Built This

I wanted to actually understand how a modern LLM works — not just call an API.

So I started from scratch and implemented every major component you'd find inside something like Llama or Mistral: the attention mechanism, the positional encoding, the optimizer, LoRA fine-tuning, RLHF with PPO, DPO, and INT4 quantization. Each piece is self-contained, readable, and comes with a test.

---

## What's Inside

The project runs as 8 sequential stages:

```
Stage 1 → Tokenizer + Dataset
Stage 2 → NumPy autograd (understand backprop without PyTorch)
Stage 3 → PyTorch training engine (device selection, seeds, checkpointing)
Stage 4 → Custom AdamW optimizer (1st/2nd moments, bias correction, weight decay)
Stage 5 → Transformer model (RoPE + Multi-Head Attention + KV-Cache + RMSNorm)
Stage 6 → Pre-training + SFT instruction tuning
Stage 7 → LoRA fine-tuning + alignment (PPO + DPO)
Stage 8 → INT4 quantization + compression benchmarks
```

Run all 8 stages at once:
```bash
python main.py
```

Or chat interactively with any checkpoint:
```bash
python interactive_chat.py
```

---

## Key Implementations

### RoPE (Rotary Positional Encoding)
