# Chapter 7: Reinforcement Learning, GRPO, and DeepSeek-R1

This chapter explains how reinforcement learning turns a DeepSeek-style base
model into a reasoning model. The code is intentionally compact: it implements
the core mechanics of GRPO with verifiable rewards rather than a full
distributed RL infrastructure.

### Code Structure

```text
ch07/01_main-chapter-code/
├── requirements.txt        # Project dependencies
├── grpo_rlvr_minimal.py    # Listings 7.1-7.4 in one runnable file
└── README.md               # How to run the chapter code
```

### How to Run

```bash
cd ch07/01_main-chapter-code
pip install -r requirements.txt
python grpo_rlvr_minimal.py
```

### What the Code Demonstrates

- `verify_math_answer` implements a deterministic RLVR reward for answer-only
  math prompts.
- `group_advantages` replaces the PPO value model with group-relative reward
  normalization.
- `grpo_loss` implements the clipped GRPO surrogate with a reference-policy KL
  penalty.
- `train_step` wires sampling, verification, advantage computation, log-prob
  scoring, and policy update into one readable training step.

This is a teaching implementation. Production GRPO training also needs robust
parsing, sequence masks, distributed rollout workers, checkpointing, and careful
monitoring of KL drift, response length, invalid outputs, and reward hacking.
