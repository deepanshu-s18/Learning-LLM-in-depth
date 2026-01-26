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
