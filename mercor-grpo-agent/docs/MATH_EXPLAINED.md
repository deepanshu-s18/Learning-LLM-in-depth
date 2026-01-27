# The Math Behind GRPO, prompt_mean, and DPPO
### Explained for Beginners — No PhD Required

---

## Part 1: What Is Reinforcement Learning for LLMs?

Imagine you're training a dog. You give it a treat when it sits on command and you ignore it (or say "no") when it doesn't. Over thousands of repetitions, the dog learns that "sit" + correct behavior → treat.

**RL for LLMs is the same idea:**
- The LLM generates text (the "action").
- The environment evaluates the text (did it fix the bug? did it answer the legal question correctly?).
- The LLM gets a reward signal: 1.0 = perfect, 0.0 = wrong.
- We update the model to make high-reward outputs more likely.

The challenge: **the model generates 1,000+ tokens per response**, and we have to decide which specific tokens to reinforce and which to push down.

---

## Part 2: The Policy Gradient (REINFORCE)

The standard formula for updating an LLM policy:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[ A \cdot \nabla_\theta \log \pi_\theta(a | s) \right]$$

In plain English:
- $\theta$ = the model's parameters (weights)
- $\pi_\theta(a | s)$ = the model's probability of generating action $a$ given state $s$
- $\log \pi_\theta$ = the **log probability** of the model's choices (what we compute in `seq_logprob`)
- $A$ = the **advantage** (how much better or worse than average was this rollout?)
- $\nabla_\theta$ = the gradient (direction to move weights)

**The key insight**: if $A > 0$, we push the model toward this output. If $A < 0$, we push away.

In code (the REINFORCE loss):
```python
loss = -(advantage * log_prob)
# Negative because we MAXIMIZE reward (gradient descent MINIMIZES loss)
```

---

## Part 3: GRPO — Group Relative Policy Optimization

**The Problem with Standard RL**: To compute advantage, you need a "value function" that estimates the expected reward from any state. Training a value function is expensive and unstable.

**GRPO's Solution**: Instead of a learned value function, use the group of rollouts themselves as the baseline.

$$A_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}}}$$

Where:
- $r_i$ = reward for rollout $i$
- $\mu_{\text{group}}$ = mean reward across the group
- $\sigma_{\text{group}}$ = standard deviation of rewards in the group

**In plain English**: 
- If rollout $i$ scored **above** the group average → $A_i > 0$ → reinforce it
- If rollout $i$ scored **below** the group average → $A_i < 0$ → push away from it
- If all rollouts got the same score → $A_i = 0$ for all → **nothing to learn** (degenerate group)

In code:
```python
rewards = torch.tensor([0.85, 0.0, 0.72, 0.0, 0.91, 0.0])
adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
# adv ≈ [+0.82, -0.97, +0.58, -0.97, +1.12, -0.97]
```

---

## Part 4: Why token_mean Breaks for Agent RL

In standard NLP training, all your examples have roughly the same length (e.g. "write a haiku" → 17 syllables). So averaging log-prob over all tokens works fine.

In agent RL, trajectories have **wildly different lengths**:
- Rollout A: 3 commands, 200 tokens, reward = 0.85 (efficient!)
- Rollout B: 25 commands, 5000 tokens, reward = 0.0 (rambling!)

With `token_mean`, the gradient from rollout B is computed as:

$$\nabla_B = A_B \cdot \frac{\sum_{t=1}^{5000} \nabla \log \pi_\theta(x_t)}{5000}$$

Even though $A_B = -0.97$ (negative, we want to push away), the raw magnitude of 5,000 gradient terms still dominates the optimization step.

Meanwhile, rollout A:

$$\nabla_A = A_A \cdot \frac{\sum_{t=1}^{200} \nabla \log \pi_\theta(x_t)}{200}$$

Only 200 terms. 25x smaller raw magnitude.

**The model ends up "learning" more from the bad rollout than the good one.**

---

## Part 5: The prompt_mean Fix [Mercor Fix #1]

The fix: normalize each rollout by its own token count FIRST, then average across the group.

$$\mathcal{L}_{\text{prompt\_mean}} = -\frac{1}{|G|} \sum_{i \in G} A_i \cdot \frac{\sum_{t=1}^{T_i} \log \pi_\theta(x_t^{(i)})}{T_i}$$

Where $T_i$ is the number of assistant tokens in rollout $i$.

**Breaking this down**:
1. For each rollout $i$: sum its log-probs, divide by **its own** token count $T_i$. This gives a per-rollout score that's independent of length.
2. Then multiply by advantage $A_i$.
3. Then average across the group (divide by $|G|$).

**Now rollout A and rollout B contribute equally**, regardless of their lengths.

```python
# Mercor code:
rollout_lp_mean = (curr_lp_per_tok * combined_mask).sum() / n_stable_tokens
loss = -(advantage * rollout_lp_mean) / group_size
```

---

## Part 6: DPPO — Decoupled Proximal Policy Optimization [Fix #3]

### The off-policy problem

In async RL (where Mercor trained 800 rollouts simultaneously), rollouts from an older policy version might still be in the training queue when the model has already been updated several times.

Applying gradients from **stale rollouts** (rollouts from an old policy) on a **different current policy** can cause catastrophic instability.

### PPO's original fix (clipping)

Standard PPO clips the importance ratio:

$$\mathcal{L}_{\text{PPO}} = \min\left( r_t A, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A \right)$$

Where $r_t = \frac{\pi_\theta(x_t)}{\pi_{\text{old}}(x_t)}$ is the importance ratio at token $t$.

PPO clips the ratio to stay within $[1-\epsilon, 1+\epsilon]$. This allows some policy drift but prevents catastrophic jumps.

### DPPO's improvement (masking instead of clipping)

DPPO from [arXiv:2602.04879] uses total variation divergence as a more principled measure:

$$\text{TV}(t) = |\exp(\log \pi_\theta(x_t) - \log \pi_{\text{old}}(x_t)) - 1|$$

Then **masks** (not clips) tokens where divergence exceeds threshold $\delta$:

$$m_t = \mathbf{1}[\text{TV}(t) < \delta]$$

$$\mathcal{L}_{\text{DPPO}} = -A \cdot \frac{\sum_t m_t \cdot \log \pi_\theta(x_t)}{\sum_t m_t}$$

**The difference from clipping**: DPPO completely **excludes** diverged tokens from the loss. PPO **clips** their contribution. Masking is safer because it avoids distorted gradients entirely.

In code:
```python
ratio = (curr_lp_per_tok - old_lp_per_tok).exp()
tv_divergence = (ratio - 1.0).abs()
dppo_mask = (tv_divergence < DPPO_DELTA).float()   # 1 = stable, 0 = masked
```

---

## Part 7: Putting It All Together

The full Mercor GRPO loss combines all three:

$$\mathcal{L} = -\frac{1}{|G|} \sum_{i \in G} A_i \cdot \frac{\sum_t m_t^{(i)} \cdot \log \pi_\theta(x_t^{(i)})}{\sum_t m_t^{(i)}}$$

Where:
- $|G|$ = group size (prompt_mean averaging across group)
- $A_i$ = group-relative advantage (GRPO baseline)
- $m_t^{(i)}$ = DPPO mask at token $t$ of rollout $i$
- Division by $\sum_t m_t^{(i)}$ = normalization by THIS rollout's stable tokens (prompt_mean)

And at the **harness level**, Fix #2 (context nudge) increases the fraction of rollouts that produce non-zero rewards, making the group statistics more meaningful.

---

## The Key Lesson

> **"Algorithm choices mattered less than the data."**
> — Mercor Research, 2026

The entire point of these 3 mathematical fixes is to get **cleaner learning signal** from the same rollouts. Better signal → better training → better model.

None of these require:
- More compute
- More data
- A bigger model

They are purely **about how you compute the loss** and **how you engineer the harness**. That's why Mercor could implement them in ~50 lines of code changes.
