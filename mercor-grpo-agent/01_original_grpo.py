"""
01_original_grpo.py
====================
THE BASELINE: Original GRPO Agent (unchanged logic, heavily annotated)

This is the exact code from swe_in_prod_vizuara_01/swe_grpo_one_step.py
with comments added to EVERY important line explaining what it does
and WHY it has limitations for multi-turn agents.

Run: python 01_original_grpo.py
"""

import json
import os
import random
import urllib.request
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────
# WHAT IS SWE-bench?
# SWE-bench is a dataset of real GitHub issues from popular Python repos
# (like Django, Flask, Numpy). Each task has:
#   - A problem_statement: the GitHub issue text
#   - A patch: the correct code fix
#   - FAIL_TO_PASS: tests that must go from failing to passing
#
# The agent must read the issue, explore the code, and write a fix.
# This is called a "software engineering agent task".
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
# ^ We use Qwen2.5-Coder-0.5B — a tiny 500M parameter coding model.
# Mercor used 397B parameters. We're doing the same math on a tiny scale
# so you can run it on any laptop.

GROUP_SIZE = 6
# ^ GRPO's key idea: instead of one rollout, run GROUP_SIZE rollouts
# of the SAME task. Compare them against each other. The better ones
# get positive advantage, the worse ones get negative advantage.

MAX_TURNS = 4
# ^ Each rollout = 4 turns (4 times the model can think + act).
# In Mercor's system, this was up to 100+ turns on complex tasks.

TEMPERATURE = 1.0
# ^ Higher temperature = more random = more diverse rollouts.
# We want diversity so different rollouts get different rewards.

LR = 1e-5
# ^ Learning rate for the gradient update.

SEED = 0

# ─────────────────────────────────────────────────────────────────────
# DOWNLOAD UTILITY FUNCTIONS
# (MockEnv, generate, first_bash_block, SYSTEM prompt — from the original repo)
# ─────────────────────────────────────────────────────────────────────
if not os.path.exists("utils.py"):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/abgoswam/swe_in_prod_vizuara_01/main/utils.py",
        "utils.py")

from utils import NO_COMMAND, SYSTEM, MockEnv, first_bash_block, generate


# ─────────────────────────────────────────────────────────────────────
# STEP 1: LOAD THE TASK
# ─────────────────────────────────────────────────────────────────────
def load_task():
    """
    Loads one task from SWE-bench_Verified.
    
    We pick the simplest task: one file changed, short patch.
    Real training would loop over thousands of these tasks.
    
    Returns:
        inst: A dict with keys: problem_statement, patch, instance_id, etc.
        fail_to_pass: List of test names that must go from FAIL → PASS.
    """
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    # Filter to tasks with only 1 file changed and a short patch (easier)
    cands = [i for i, r in enumerate(ds)
             if r["patch"].count("diff --git") == 1 and len(r["patch"]) < 1800]
    inst = ds[cands[0]]
    return inst, json.loads(inst["FAIL_TO_PASS"])


# ─────────────────────────────────────────────────────────────────────
# STEP 2: LOAD THE POLICY (the model)
# ─────────────────────────────────────────────────────────────────────
def load_policy(device):
    """
    CONCEPT: In RL, the "policy" is the brain that makes decisions.
    Here, the policy is the LLM (Qwen2.5-Coder-0.5B).
    
    We load it WITHOUT LoRA adapters yet. We don't add trainable
    parameters until AFTER we collect all rollouts. This is because:
    - Generation (rollouts) needs fast inference → no extra parameters
    - Training needs gradients → add LoRA adapters just before update
    
    This separation is called "actor-critic decoupling" and is standard
    in production RL (also how SkyRL/vLLM separate rollout from training).
    """
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token   # some models need this

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa"   # scaled dot product attention (fast)
    ).to(device)

    print(f"{MODEL}\n{sum(p.numel() for p in model.parameters()):,} parameters, none trainable yet")
    return model, tok


# ─────────────────────────────────────────────────────────────────────
# STEP 3: RUN THE AGENT (collect one rollout/trajectory)
# ─────────────────────────────────────────────────────────────────────
def run_agent(model, tok, inst, fail_to_pass, max_turns=MAX_TURNS,
              temperature=TEMPERATURE, sample=generate):
    """
    CONCEPT: The "harness" = the environment the agent acts in.
    
    In Mercor's system, the harness was a Docker container with:
      - A simulated company file system
      - PDF/Excel/Slides MCP servers
      - Email and Slack servers
    
    Here, the harness is MockEnv — a simple simulated bash terminal.
    
    THE AGENT LOOP (ReAct pattern):
    For each turn:
      1. Show model the current context (system prompt + issue + history)
      2. Model generates a reply (thinks + writes bash command)
      3. We extract the bash command from the reply
      4. Run the command in MockEnv (fake terminal)
      5. Append model output + terminal result to context
      6. Repeat until max_turns hit
    
    ⚠️  LIMITATION IN THIS BASELINE:
    There is NO warning to the model that time is running out.
    The model will often still be "exploring" when MAX_TURNS is hit.
    When there's no patch → reward = 0.
    This wastes rollouts and distorts the reward signal.
    (Mercor's "Context Nudge" fix addresses this — see 02_mercor_grpo.py)
    """
    env = MockEnv(fail_to_pass)
    context = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"ISSUE:\n{inst['problem_statement'][:1500]}"}
    ]

    for turn_idx in range(max_turns):
        # Build the prompt by applying the chat template
        prompt = tok.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
        
        # Let the model generate a response (this is the "action")
        reply  = sample(model, tok, prompt, temperature=temperature)
        
        # Parse the bash command out of the model's reply
        action = first_bash_block(reply)
        
        # Execute the command in the simulated environment
        obs = env.run(action) if action else NO_COMMAND
        
        # Append both sides to the context for next turn
        context += [
            {"role": "assistant", "content": reply},
            {"role": "user",      "content": obs[:800]}   # truncate long outputs
        ]
        
        # ⚠️  BASELINE GAP: No nudge injected here even when turn_idx == max_turns - 1
        # The model doesn't know it's the last turn.
        # Result: many rollouts end with reward = 0 because no patch was written.

    return dict(messages=context, patch=env.patch(), final=dict(env.fs), calls=env.calls)


# ─────────────────────────────────────────────────────────────────────
# STEP 4: THE REWARD FUNCTION (Verifier)
# ─────────────────────────────────────────────────────────────────────
def reward_random(patch, rng):
    """
    CONCEPT: The reward function answers "how good was this rollout?"
    
    In Mercor's system, the reward was computed by an LLM judge
    that read the final output and graded it against the task criteria.
    At 800 concurrent rollouts, this required round-robin API key rotation.
    
    Here we use a simple stand-in:
      - No patch written → reward = 0.0
      - Patch written → random score in [0, 1]
    
    In real training, you'd replace this with:
      - Unit test execution (SWE-bench)
      - LLM judge (Mercor APEX-Agents)
      - Triton kernel performance measurement (your triton-rl project)
    """
    if not patch:
        return 0.0
    return round(float(rng.random()), 3)


# ─────────────────────────────────────────────────────────────────────
# STEP 5: ADD LoRA ADAPTERS (trainable parameters)
# ─────────────────────────────────────────────────────────────────────
def add_lora(model):
    """
    CONCEPT: LoRA = Low-Rank Adaptation.
    
    We don't train ALL 500M parameters. That would:
    1. Require huge GPU memory
    2. Forget everything the model learned in pretraining
    
    Instead, we freeze the original weights and add tiny "adapter" 
    matrices (rank-8) to the attention layers. Only these adapters 
    are updated. This is called Parameter-Efficient Fine-Tuning (PEFT).
    
    In Mercor's 397B run, they used Megatron-LM with tensor parallelism
    across 8 training GPUs. We use PEFT LoRA on a single GPU.
    """
    model = get_peft_model(model, LoraConfig(
        r=8,           # rank of the adapter matrix (low = memory efficient)
        lora_alpha=16, # scaling factor (usually 2x rank)
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # attention layers
    ))
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────
# STEP 6: BUILD MASKED TOKEN IDS
# ─────────────────────────────────────────────────────────────────────
def build_masked(messages, tokenizer, max_len=3072):
    """
    CONCEPT: In RL, we only want to update the model on ASSISTANT tokens.
    The user/system tokens are the "observation" — we didn't generate them.
    
    We do this by:
    1. Tokenizing the full conversation
    2. Setting labels to -100 for user/system tokens (these are ignored in loss)
    3. Keeping the actual token IDs for assistant turns
    
    This is called "supervised token masking" and is standard in all
    LLM RL frameworks (TRL, SkyRL, Unsloth).
    
    Example:
        [SYSTEM] help me fix...    → labels: [-100, -100, -100, ...]
        [ASSISTANT] cat file.py    → labels: [token_id_1, token_id_2, ...]
        [USER] output here...      → labels: [-100, -100, -100, ...]
        [ASSISTANT] patch: ...     → labels: [token_id_3, token_id_4, ...]
    """
    ids, labels, prev = [], [], ""
    for i, m in enumerate(messages):
        cur = tokenizer.apply_chat_template(messages[:i + 1], tokenize=False)
        assert cur.startswith(prev), "chat template is not append-only"
        seg = tokenizer(cur[len(prev):], add_special_tokens=False)["input_ids"]
        ids += seg
        labels += seg if m["role"] == "assistant" else [-100] * len(seg)
        prev = cur
    return ids[:max_len], labels[:max_len]


# ─────────────────────────────────────────────────────────────────────
# STEP 7: COMPUTE SEQUENCE LOG PROBABILITY
# ─────────────────────────────────────────────────────────────────────
def seq_logprob(model, tok, messages, device):
    """
    CONCEPT: Log probability = "how confident is the model in what it said?"
    
    Given a complete conversation (messages), this function computes:
    - The log probability of every assistant token in the conversation
    - The MEAN across all assistant tokens (called "token_mean")
    
    ⚠️  THIS IS THE BASELINE'S BIGGEST FLAW:
    Token_mean divides by the TOTAL number of assistant tokens.
    A 5000-token rambling rollout contributes 25x more to the 
    gradient than a 200-token efficient rollout.
    
    Mercor's fix: "prompt_mean" — normalize per rollout, then average.
    See seq_logprob_per_token() in 02_mercor_grpo.py.
    
    Returns:
        mean_logprob: scalar tensor (used in REINFORCE loss)
        n_sup_tokens: int (number of assistant tokens, for debugging)
    """
    ids, labs = build_masked(messages, tok)
    
    # Convert to tensors
    t = torch.tensor([ids], device=device)
    
    # Create mask: 1.0 for assistant tokens, 0.0 for user/system tokens
    msk = torch.tensor([[0. if l == -100 else 1. for l in labs]], device=device)[:, 1:]
    
    # Forward pass through model
    logits = model(t).logits[:, :-1]
    
    # Convert logits → log probabilities for the ACTUAL token choices
    lp = torch.log_softmax(logits.float(), -1).gather(-1, t[:, 1:].unsqueeze(-1)).squeeze(-1)
    
    # ⚠️  token_mean: sum of logprobs / total_tokens (biased toward long rollouts!)
    return (lp * msk).sum() / msk.sum().clamp(min=1), msk.sum().item()


# ─────────────────────────────────────────────────────────────────────
# STEP 8: THE GRPO UPDATE (the actual training)
# ─────────────────────────────────────────────────────────────────────
def grpo_step(model, tok, group, reward, device, lr=LR):
    """
    CONCEPT: GRPO = Group Relative Policy Optimization
    
    Unlike standard RL (where you compare against a value function),
    GRPO compares rollouts of the SAME TASK against each other.
    
    Step 1: Compute GROUP ADVANTAGE
        advantage_i = (reward_i - mean_reward) / std_reward
        
        This normalizes rewards so:
        - Rollouts better than average get POSITIVE advantage (+)
        - Rollouts worse than average get NEGATIVE advantage (-)
    
    Step 2: REINFORCE loss
        loss = -sum(advantage_i * log_prob_i) / group_size
        
        If advantage > 0: increase log_prob (do this more often)
        If advantage < 0: decrease log_prob (do this less often)
    
    ⚠️  BASELINE FLAWS (fixed in 02_mercor_grpo.py):
    1. token_mean in seq_logprob = length-biased gradients
    2. No DPPO clipping = policy can drift wildly
    3. No context nudge = many 0-reward rollouts from hitting turn limit
    """
    rewards = torch.tensor(reward, dtype=torch.float)
    
    # ────────────────────────────────────────────────
    # COMPUTE ADVANTAGES
    # ────────────────────────────────────────────────
    adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
    # ^ 1e-4 prevents division by zero when all rewards are identical

    print(f"\n--- GRPO STEP (BASELINE) ---")
    print(f"mean reward {rewards.mean():.3f}   std {rewards.std(unbiased=False):.3f}")
    
    # Warn if all rollouts scored the same (no learning signal)
    if rewards.std(unbiased=False) < 1e-8:
        print(f"\n  ⚠️  DEGENERATE GROUP: All rewards identical → all advantages = 0 → no learning!")
        print(f"  This happens often with tiny models. Increase GROUP_SIZE or temperature.\n")
    
    # Show the breakdown for debugging
    sup_tokens = [int(seq_logprob(model, tok, g["messages"], device)[1]) for g in group]
    print("\nRollout breakdown:")
    for i, (r, a, t) in enumerate(zip(rewards, adv, sup_tokens)):
        print(f"  rollout {i}: reward={r:.3f}  advantage={a:.3f}  tokens={t}")
    
    # ────────────────────────────────────────────────
    # GRADIENT CONTRIBUTION WARNING
    # ────────────────────────────────────────────────
    max_tok = max(sup_tokens)
    min_tok = min(sup_tokens)
    if max_tok > 0 and min_tok > 0:
        ratio = max_tok / min_tok
        print(f"\n⚠️  BASELINE LENGTH BIAS: Longest rollout ({max_tok} tokens) contributes "
              f"{ratio:.1f}x more to gradient than shortest ({min_tok} tokens)!")
        print("    This is what prompt_mean fixes in the Mercor paper.")

    # ────────────────────────────────────────────────
    # THE REINFORCE UPDATE
    # ────────────────────────────────────────────────
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    logp_before = [seq_logprob(model, tok, g["messages"], device)[0].item() for g in group]

    model.train()
    opt.zero_grad(set_to_none=True)
    loss_total = 0.0
    
    for g, a in zip(group, adv):
        lp, _ = seq_logprob(model, tok, g["messages"], device)
        
        # ⚠️  BASELINE LOSS: token_mean aggregation (length-biased!)
        # Long rollouts with zero advantage still contribute more gradient noise
        loss = -(a.to(device) * lp) / len(group)
        loss.backward()
        loss_total += loss.item()

    # Clip gradients to prevent exploding gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()

    logp_after = [seq_logprob(model, tok, g["messages"], device)[0].item() for g in group]

    print(f"\nloss {loss_total:+.5f}   grad_norm {grad_norm:.4f}")
    print("\nLogprob changes after update:")
    for i, (a, b, c) in enumerate(zip(adv, logp_before, logp_after)):
        direction = "↑" if c > b else "↓"
        print(f"  rollout {i}: advantage={a:.3f}  {b:.4f} → {c:.4f}  {direction}  (Δ={c-b:.5f})")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BASELINE GRPO (Original — with annotated limitations)")
    print("=" * 60)
    
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. Load one SWE-bench task
    inst, fail_to_pass = load_task()
    print(f"\nTask: {inst['instance_id']}  |  {len(fail_to_pass)} tests must go FAIL → PASS")
    print(f"Issue preview: {inst['problem_statement'][:200]}...")

    # 2. Load the model (no LoRA yet)
    model, tok = load_policy(device)

    # 3. Collect GROUP_SIZE rollouts
    rng = np.random.default_rng(SEED)
    group, reward = [], []
    
    print(f"\nCollecting {GROUP_SIZE} rollouts (agent attempts at fixing the bug)...")
    for i in range(GROUP_SIZE):
        rollout = run_agent(model, tok, inst, fail_to_pass)
        score = reward_random(rollout["patch"], rng)
        group.append(rollout)
        reward.append(score)
        n_commands = len(rollout["calls"])
        n_tokens = sum(len(m["content"].split()) for m in rollout["messages"])
        print(f"  rollout {i}: {n_commands} commands  "
              f"patch={'YES' if rollout['patch'] else 'NO'}  "
              f"~{n_tokens} words  "
              f"reward={score}")

    reward = np.array(reward)

    # 4. Add LoRA adapters (trainable params added NOW, not before rollouts)
    print(f"\nAdding LoRA adapters for training...")
    model = add_lora(model)

    # 5. One GRPO gradient step
    grpo_step(model, tok, group, reward, device)

    print("\n✅ Baseline GRPO complete.")
    print("   Run 02_mercor_grpo.py to see the upgrades.")


if __name__ == "__main__":
    main()
