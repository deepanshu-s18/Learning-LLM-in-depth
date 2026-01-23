# Mercor-Style GRPO Agent Upgrades
### From the paper: *"Training Frontier Knowledge Work Agents: A 397B RL Training Guide with SkyRL"*
> **Published**: September 1, 2026 by Mercor Research + SkyRL (UC Berkeley)  
> **Original**: [mercor.com/blog/training-frontier-knowledge-work-agents](https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/)  
> **Code**: [github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe)

[![Paper](https://img.shields.io/badge/Paper-Mercor%20%2B%20SkyRL%202026-blue)](https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/)
[![GRPO](https://img.shields.io/badge/Algorithm-GRPO%20%2B%20DPPO-orange)](https://arxiv.org/abs/2402.03300)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE)

---

## What is this repo?

This repo implements the **3 core algorithm upgrades** from the Mercor + SkyRL paper on top of a standard GRPO agent baseline. You can run `03_comparison_runner.py` to see both versions side-by-side on a live model (Qwen2.5-Coder-0.5B) in under 5 minutes.

**The purpose:**
1. Understand *why* standard GRPO breaks for multi-turn agents.
2. Implement the exact 3 fixes Mercor used to train a 397B-parameter model.
3. Compare old vs new side-by-side with a working benchmark script.

---

## What did Mercor actually DO?

Mercor took an open-source LLM (Qwen3.5-397B), and used **Reinforcement Learning (RL)** to train it to do complex office work:
- Read PDFs with 50 pages.
- Search through emails.
