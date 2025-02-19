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
