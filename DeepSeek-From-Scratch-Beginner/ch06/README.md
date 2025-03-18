# Chapter 6: The DeepSeek Training Pipeline

This chapter brings together all the architectural innovations from previous chapters into a complete, trainable "MiniDeepSeek V3" model. Unlike other chapters that use Jupyter notebooks, this chapter consists of **four standalone Python scripts** that form a complete training pipeline.

### Code Structure

The code is organized as a pipeline — run the scripts in order:

```
ch06/01_main-chapter-code/
├── requirements.txt    # Project dependencies (Listing 6.1)
├── prepare.py          # Data preparation pipeline (Listings 6.2–6.5)
├── model.py            # Complete MiniDeepSeek architecture (Listings 6.6–6.18)
├── train.py            # Training loop with evaluation (Listings 6.19–6.24)
└── sample.py           # Text generation from trained model
```

### How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and tokenize the TinyStories dataset
