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
Relative-distance invariant positional encoding — the same technique used in Llama, Mistral, and most modern LLMs.

### LoRA (Low-Rank Adaptation)
Fine-tunes models with 10–100x fewer parameters. Formula: `W₀ + (α/r)·BA`

### DPO (Direct Preference Optimization)
Trains the model to prefer good responses over bad ones directly on log-probabilities — no reward model needed.

### PPO Alignment
Full actor-critic alignment loop with KL divergence penalty against a frozen reference policy.

### INT4 Quantization
Affine scale/zero-point calibration + nibble packing. Measures reconstruction MSE and compression ratio.

### KV-Cache
Side-by-side latency and throughput benchmark: cached vs uncached generation at increasing context lengths.

---

## Project Structure

```
llm-lite/
├── main.py                    # Master orchestrator — runs all 8 stages
├── interactive_chat.py        # Live chat with any checkpoint
├── data/
│   └── dataset.py             # Tokenizer, pretraining corpus, SFT pairs, preference triples
├── src/
│   ├── foundations/
│   │   ├── numpy_autograd.py  # Pure NumPy backprop, verified against PyTorch
│   │   └── pytorch_engine.py  # Device selector, seeding, gradient clipping, checkpointing
│   ├── optimization/
│   │   └── custom_adamw.py    # AdamW from scratch, verified against torch.optim.AdamW
│   ├── model/
│   │   ├── rope.py            # Rotary Position Embedding
│   │   ├── attention.py       # Multi-Head Attention with RoPE + KV-Cache
│   │   ├── transformer.py     # Full decoder Transformer (RMSNorm, GELU, causal LM head)
│   │   └── kv_cache.py        # KV-Cache benchmark harness
│   ├── tuning/
│   │   ├── pretrain.py        # Causal LM pre-training with perplexity tracking
│   │   ├── sft.py             # Instruction fine-tuning with prompt loss masking
│   │   └── lora.py            # LoRALinear layer + weight merging
│   ├── alignment/
│   │   ├── reward_model.py    # Bradley-Terry pairwise preference loss
│   │   ├── ppo_aligner.py     # PPO with KL penalty
│   │   └── dpo_aligner.py     # Direct Preference Optimization
│   └── compression/
│       └── quantizer.py       # INT8 + INT4 nibble packing + compression metrics
└── tests/
    └── test_all_modules.py    # Unit tests across all 8 stages
```

---

## Run It

```bash
pip install torch numpy

# Run everything end-to-end
python main.py

# Interactive chat
python interactive_chat.py

# Run unit tests
python -m unittest discover -s tests -p "test_*.py"
```

---

## What I Learned

Before this, I understood LLMs conceptually. After this, I understand them mechanically.

The part that surprised me most: DPO is simpler and more stable than PPO in practice. PPO needs a value network, a reward model, and careful KL tuning. DPO bypasses all of that and works directly on the policy log-probabilities. The math is cleaner and the training is more predictable.

INT4 quantization was the other revelation — you can cut model size by 75% and reconstruction MSE stays surprisingly low if your calibration dataset is representative.

---

*Built by [Shobhit Agnihotri](https://github.com/shobhitagnihotri69)*
