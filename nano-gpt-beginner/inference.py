import torch
import torch.nn.functional as F
from architecture.tokenizer import get_tokenizer

context_len = 4000
tokenizer   = get_tokenizer()

def generate_text(model, prompt, max_tokens=100, temperature=0.9, top_k=40):
    device = next(model.parameters()).device
    model.eval()
    idx = torch.tensor(tokenizer.encode(prompt), device=device)
    for _ in range(max_tokens):
        cond   = idx[-context_len:]
        with torch.inference_mode():
            logits = model(cond.unsqueeze(0))[0, -1] / temperature
