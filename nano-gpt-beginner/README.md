# nano-gpt-beginner

A beginner-friendly implementation of NanoGPT built from scratch in PyTorch.

A from-scratch implementation of NanoGPT, rewritten to be clean and easy to read.

## Architecture

- **GPT-2 style model** (`architecture/gpt2.py`) — standard transformer with multi-head attention
- **GPT-OSS model** (`architecture/gptoss.py`) — grouped query attention (GQA) + Mixture of Experts (MoE)
- **Tokenizer** (`architecture/tokenizer.py`) — custom tokenizer wrapper

## Training

```bash
pip install -r requirements.txt
python train.py
```

## Inference

```python
from inference import generate_text
from architecture.gptoss import Transformer, ModelConfig
import torch

model = Transformer(ModelConfig())
text  = generate_text(model, "Once upon a time")
print(text)
```

## Files

| File | Description |
|------|-------------|
| `train.py` | Main training entry point |
| `inference.py` | Text generation |
| `architecture/gpt2.py` | GPT-2 model |
| `architecture/gptoss.py` | GPT-OSS with GQA + MoE |
| `training/trainer.py` | Training loop |
| `training/data_loader.py` | TinyStories data loading |
