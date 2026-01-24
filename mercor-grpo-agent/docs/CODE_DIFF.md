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
