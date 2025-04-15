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
