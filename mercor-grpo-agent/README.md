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

### Fix 2: Context Nudge — Harness Engineering (+3.0 points in the paper)

**The problem:**

In the original `run_agent()` function, the agent just runs until it hits `MAX_TURNS`. If the task is complex, the model might still be in the middle of exploring when it suddenly gets cut off. The trajectory ends, the reward is 0 (no patch produced), and the model gets punished for running out of turns.

This is not a *model failure*. It's a *harness failure*. The model never knew time was running out.

**The Mercor fix:**

Inject a warning message when 80% of the turn budget is consumed:
```python
# At turn 3 out of 4 (MAX_TURNS=4), inject this:
context.append({
    "role": "user",
    "content": "[SYSTEM ALERT]: You have 1 turn remaining. "
               "Stop exploring. Produce your FINAL patch immediately."
})
```

This bought Mercor **+3.0 points** with zero training cost. It's a pure harness fix.

---

### Fix 3: DPPO — Decoupled Proximal Policy Optimization

**The problem:**

In standard GRPO (and the original `swe_grpo_one_step.py`), we use REINFORCE:
```python
loss = -(advantage * log_prob)
```

When the model is updated across multiple gradient steps or in async training (where rollouts arrive from an older policy version), the current policy might have **drifted far** from the rollout policy. Updating on stale rollouts can destabilize training.

**The Mercor fix (DPPO from paper [8]):**

Calculate the **token-level importance ratio** between current and old policy:

$$r_t = \exp(\log \pi_\theta(x_t) - \log \pi_{\text{old}}(x_t))$$

Then mask out (zero out) tokens where the divergence exceeds a threshold $\delta$:

```python
# If the current policy has drifted too far from rollout policy, skip this token
ratio = (current_logprob - rollout_logprob).exp()
total_variation_mask = (ratio - 1).abs() < delta   # keeps only stable tokens
loss = -(advantage * token_logprobs * total_variation_mask).sum() / mask.sum()
```

This prevents catastrophic divergence during multi-turn agent training.

---

## File Structure

```
mercor-grpo-agent/
│
├── README.md                        ← You are here. The full research explanation.
│
├── 01_original_grpo.py              ← The original GRPO code (baseline, unchanged)
│                                       with heavy comments explaining what each line does.
│
├── 02_mercor_grpo.py                ← The upgraded code with all 3 Mercor fixes.
│                                       Every change is annotated with a [MERCOR FIX #N] comment.
│
├── 03_comparison_runner.py          ← Run both versions side-by-side and print a comparison table.
│
├── docs/
│   ├── PAPER_BREAKDOWN.md           ← Full plain-English breakdown of the Mercor paper.
│   ├── CODE_DIFF.md                 ← Line-by-line diff of what changed and WHY.
│   └── MATH_EXPLAINED.md           ← The math behind GRPO, prompt_mean, and DPPO.
│
└── requirements.txt                 ← All dependencies to run this.
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the original (baseline) GRPO
python 01_original_grpo.py

# 3. Run the upgraded Mercor-style GRPO
python 02_mercor_grpo.py

# 4. Run both and compare (shows which is more stable)
python 03_comparison_runner.py
```

---

## What You Will See

When you run `03_comparison_runner.py`, you'll see output like:

```
============================================================
             GRPO COMPARISON: BASELINE vs MERCOR
============================================================

Generating 6 rollouts from Qwen2.5-Coder-0.5B-Instruct...

  rollout  reward  tokens  advantage (baseline)  advantage (mercor)
        0    0.85     200               +1.234              +1.234
        1    0.00    3500               -0.823              -0.823
        2    0.72     180               +0.912              +0.912
        3    0.00     800               -0.823              -0.823
        4    0.91     350               +1.012              +1.012
        5    0.00    5000               -0.823              -0.823

