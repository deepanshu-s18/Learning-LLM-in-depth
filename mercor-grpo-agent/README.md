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
- Write PowerPoint presentations.
- Do legal research.

They did this using a method called **GRPO** (Group Relative Policy Optimization), which is a variant of the famous **PPO** algorithm that powers ChatGPT.

Their results:
- **+70% relative improvement** in task success rate on the APEX-Agents benchmark.
- The model went from solving 16% of knowledge work tasks to solving **27%**.

All of that came from just **3 surgical fixes** to the standard RL algorithm.

---

## The Problem: Why Standard GRPO Breaks for Agents

Imagine you're training an agent on SWE-bench (GitHub bug-fix tasks). Your agent runs 6 different rollouts (attempts) at fixing the same bug, and they look like this:

```
Rollout 0: Wrote 3 bash commands, patched the file → 200 tokens total, reward = 0.85 ✅
Rollout 1: Rambled for 15 commands, wandered around, gave up → 3,500 tokens total, reward = 0.0 ❌
Rollout 2: Wrote 2 bash commands, patched the file → 180 tokens total, reward = 0.72 ✅
Rollout 3: Ran out of turns, no patch produced → 800 tokens total, reward = 0.0 ❌
Rollout 4: Explored correctly, patched → 350 tokens total, reward = 0.91 ✅
Rollout 5: Rambled for 20 commands → 5,000 tokens total, reward = 0.0 ❌
```

With standard GRPO, the **gradient is dominated by the rambling rollouts** because they have the most tokens. The model "learns" from rollouts 1, 3, and 5 more than from rollouts 0, 2, and 4. That's backwards! The model should learn most from the SHORT, SUCCESSFUL rollouts.

This is the fundamental problem Mercor identified and solved.

---

## The 3 Mercor Fixes (What We Implement)

### Fix 1: `prompt_mean` Token Aggregation (+3.9 points in the paper)

**The problem with the old code (line 165 of `swe_grpo_one_step.py`):**
```python
# OLD: "token_mean" — the long rambling rollouts dominate everything
loss = -(a.to(device) * lp) / len(group)
```

When `lp` (log probability) is computed in `seq_logprob()`, it's the **sum of all token log probs divided by number of tokens**. A 5,000-token rollout with zero advantage contributes 10x more gradient noise than a 500-token successful rollout.

**The Mercor fix:**
```python
# NEW: "prompt_mean" — every rollout counts equally regardless of length
# Step 1: Compute per-token loss (NOT averaged yet)
# Step 2: Sum over each rollout's tokens
# Step 3: Divide by that rollout's OWN token count (normalize within)
# Step 4: Average across the group
loss = -sum(advantage_i * (sum_of_token_logprobs_i / n_tokens_i) for each rollout) / group_size
```

This is called `prompt_mean` because we normalize by prompt group, not global token count.

---

