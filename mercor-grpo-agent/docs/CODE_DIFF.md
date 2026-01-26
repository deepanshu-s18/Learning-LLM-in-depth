# Code Changes Explained: Baseline vs. Mercor GRPO
### Line-by-line: What Changed and WHY

This file explains every single code change between `01_original_grpo.py` (baseline) and `02_mercor_grpo.py` (Mercor upgraded). It is written for someone who understands Python but has never read an RL paper.

---

## Change 1: `run_agent()` → `run_agent_with_nudge()` [Fix #2]

### Baseline code (lines 80–86 of 01_original_grpo.py):
```python
for turn_idx in range(max_turns):
    prompt = tok.apply_chat_template(context, ...)
    reply  = generate(model, tok, prompt, ...)
    action = first_bash_block(reply)
    obs    = env.run(action) if action else NO_COMMAND
    context += [{"role": "assistant", "content": reply},
                {"role": "user",      "content": obs[:800]}]
```
**The problem**: The model never knows it's on its last turn. So on turn 4/4, it might still be running `ls` to explore the codebase instead of writing the final patch. The turn limit hits → no patch → reward = 0.

### Mercor code (added block inside the for loop):
```python
for turn_idx in range(max_turns):
    
    # ─── NEW: Context Nudge ───────────────────────────────────
    if turn_idx == max_turns - 1:              # ← Is this the last turn?
        context[-1]["content"] += (
            "\n\n⚠️ [SYSTEM]: Final turn. Write your complete fix NOW."
        )
    # ─────────────────────────────────────────────────────────
    
    prompt = tok.apply_chat_template(context, ...)
    reply  = generate(model, tok, prompt, ...)
    ...
```
**What changed**: One `if` block, 3 lines. That's it.

**Why it gives +3.0 points**: On long complex tasks (Mercor uses 50–100 turn agents for legal/banking tasks), the model regularly runs out of turns without producing output. Injecting the warning saves those rollouts from getting zero reward.

> **Quote from Mercor paper**: *"Fewer rollouts blow the context, so fewer get zeroed across the board, hence more usable signal per batch."*

---

## Change 2: `seq_logprob()` → `seq_logprob_per_token()` [Needed for Fixes #1 and #3]

### Baseline code:
```python
def seq_logprob(model, tok, messages, device):
    ids, labs = build_masked(messages, tok)
    t = torch.tensor([ids], device=device)
    msk = ...
    logits = model(t).logits[:, :-1]
    lp = torch.log_softmax(logits.float(), -1).gather(...)
    
    return (lp * msk).sum() / msk.sum().clamp(min=1), msk.sum().item()
    #                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                This is the MEAN over ALL tokens (token_mean)
    #                Returns ONE scalar number
```

### Mercor code (new function):
```python
def seq_logprob_per_token(model, tok, messages, device):
    ids, labs = build_masked(messages, tok)
    t = torch.tensor([ids], device=device)
    msk = ...
    logits = model(t).logits[:, :-1]
    lp = torch.log_softmax(logits.float(), -1).gather(...)
    
    return lp.squeeze(0), msk.squeeze(0), int(msk.sum().item())
    #       ^^^^^^^^^^^^  ^^^^^^^^^^^^^^
    #       TENSOR of logprobs  TENSOR of mask
    #       (one value per token, NOT averaged yet)
```

**What changed**: Instead of averaging immediately, we return the raw per-token logprobs and the mask separately.

**Why we need this**:
- **Fix #1 (prompt_mean)**: To normalize by each rollout's own token count, we need each rollout's tokens separate. If we average globally first, we lose that information.
- **Fix #3 (DPPO)**: To compute divergence at each token, we need to compare current policy vs. old policy at each token position. We can't do that if we've already averaged everything into one number.

---

## Change 3: The Loss Function [Fix #1 — prompt_mean]

### Baseline loss (line 165 of 01_original_grpo.py):
```python
for g, a in zip(group, adv):
    lp, _ = seq_logprob(model, tok, g["messages"], device)
    
    loss = -(a.to(device) * lp) / len(group)
    #                         ^
    #                         lp is already token_mean (averaged globally)
    loss.backward()
```

**What happens mathematically**:

Rollout A has 200 tokens with reward 0.85 → `lp_A = sum_of_200_tokens / 200`  
Rollout B has 5000 tokens with reward 0.0 → `lp_B = sum_of_5000_tokens / 5000`

When we compute gradients: rollout B's 5000 tokens each push gradients individually. Rollout A's 200 tokens each push individually. So B contributes 25x more raw gradient magnitude to the optimizer.

Even if B has zero advantage (adv=0), it still contributes **gradient noise** to the optimization step.

### Mercor loss (from 02_mercor_grpo.py):
```python
for i, (g, a, old_lp_per_tok) in enumerate(zip(group, adv, rollout_logprobs)):
    
    curr_lp_per_tok, mask, n_tokens = seq_logprob_per_token(model, tok, g["messages"], device)
    
    # ... (DPPO masking) ...
    combined_mask = mask * dppo_mask
    n_stable = combined_mask.sum()
    
    # ── [FIX #1: prompt_mean] ─────────────────────────────────────────
    rollout_lp_sum  = (curr_lp_per_tok * combined_mask).sum()
    rollout_lp_mean = rollout_lp_sum / n_stable    # ← Normalize by THIS rollout's tokens
    
    loss = -(a.to(device) * rollout_lp_mean) / group_size   # ← Divide by group size
    # ─────────────────────────────────────────────────────────────────
    loss.backward()
```

**What changes mathematically**:

Rollout A: `loss_A = -adv_A * (sum_200_tokens / 200) / 6`  
Rollout B: `loss_B = -adv_B * (sum_5000_tokens / 5000) / 6`

Each rollout contributes **exactly 1/6** of the total gradient. Whether it has 200 tokens or 50,000 tokens, it counts the same. **Length is no longer a confound.**

---

## Change 4: DPPO Token Masking [Fix #3]

This is entirely new code added inside `compute_mercor_loss()`. Nothing from the baseline does this.

### The new block:
```python
# Before the training loop: snapshot old logprobs
with torch.no_grad():
    for g in group:
        lp_tok, _, _ = seq_logprob_per_token(model, tok, g["messages"], device)
        rollout_logprobs.append(lp_tok.detach().cpu())   # freeze old policy snapshot

# During training:
ratio = (curr_lp_per_tok - old_lp_per_tok).exp()
#         ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
#         current policy    old policy (from snapshot above)
#
# ratio > 1 means current policy became more confident on this token
# ratio < 1 means current policy became less confident
# ratio = 1 means no change

tv_divergence = (ratio - 1.0).abs()
dppo_mask = (tv_divergence < DPPO_DELTA).float()
# ^ 1 where token is stable, 0 where policy drifted too much

combined_mask = mask * dppo_mask   # only stable assistant tokens
```

**In plain English**:

Imagine you run 6 rollouts, and while you're training on rollout 1, the model's weights change slightly. By the time you train on rollout 6, the model is a little different from the model that generated those rollouts.

Some tokens in rollout 6 might have been generated when the model was very confident (`ratio >> 1`). If the model has since changed (`ratio << 1`), applying gradients from those tokens is **stale** — you're updating based on information that's no longer accurate about the current policy.

DPPO says: "If a token's importance ratio deviates more than 20% from 1.0, don't update the policy on that token." This prevents instability.

---

## Summary of Changes

| File | Lines Changed | What | Gain |
|---|---|---|---|
| `run_agent()` | +3 lines | If last turn, append nudge to user message | +3.0 pts |
| `seq_logprob()` | Full replacement | Return per-token tensors instead of mean scalar | Enables Fix #1 and #3 |
| `grpo_step()` loss | ~10 lines | Normalize per-rollout, not globally | +3.9 pts |
| New `compute_mercor_loss()` | ~30 lines | DPPO ratio + mask before loss | Stability |

**Total lines of new code: ~50 lines across both files.**

That's it. 50 lines of code. That's what gave Mercor +3.9 + 3.0 = +6.9 measurable points on a benchmark and the basis for a successful 397B model training run.
