import torch
import torch.nn.functional as F
from architecture.tokenizer import get_tokenizer

context_len = 4000
tokenizer   = get_tokenizer()

def generate_text(model, prompt, max_tokens=100, temperature=0.9, top_k=40):
    device = next(model.parameters()).device
    model.eval()
