# Chapter 7 Main Code: Minimal GRPO + RLVR

This folder contains the runnable code corresponding to the Chapter 7 listings.
It is deliberately small so that the algorithmic pieces remain visible.

```bash
pip install -r requirements.txt
python grpo_rlvr_minimal.py
```

The demo uses tiny tensor inputs for the loss path and a simple deterministic
math-answer verifier for the reward path. Replace the placeholder
`sample_group` and `sequence_logprobs` functions with calls to your policy model
when adapting this to a real language model.
