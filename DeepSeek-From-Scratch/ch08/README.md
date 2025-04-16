# Chapter 8: Knowledge Distillation — Making Powerful Models Practical

This chapter covers knowledge distillation, the technique that compresses the capabilities of massive models into small, deployable ones. DeepSeek-R1-Distill-Qwen-1.5B (0.2% of the teacher's size) beats GPT-4o on mathematical reasoning — this chapter explains how.

### Main Chapter Code

- [Chapter_8.ipynb](01_main-chapter-code/Chapter_8.ipynb) — Contains:
  - **Listing 8.1**: Temperature-scaled softmax visualization — seeing the dark knowledge emerge
  - **Listing 8.2**: The complete knowledge distillation loss function (hard labels + soft targets with T² scaling)
  - **Listing 8.3**: Teacher and student CNN architectures + `DistillationTrainer` class
  - **Listings 8.4 & 8.5**: Full distillation pipeline on CIFAR-10 with evaluation and comparison

### Key Concepts

- **Dark Knowledge**: The information encoded in the relative probabilities of incorrect classes
- **Temperature Scaling**: Raising T from 1 to 3–5 reveals inter-class relationships hidden in peaked distributions
- **Distillation Loss**: `L = α · L_hard + (1-α) · T² · KL(teacher_soft || student_soft)`
- **DeepSeek's Approach**: Chain-of-thought distillation via SFT on 800K curated teacher outputs (not classical logit-matching)

### Relevant Videos for this Chapter

- [Knowledge Distillation Explained](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSiOpKKlHCyOq9lnp-dLvlms)
