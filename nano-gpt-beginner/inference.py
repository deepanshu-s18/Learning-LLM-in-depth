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
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float("-inf")
        idx_next = torch.multinomial(F.softmax(logits, dim=-1), 1)
        idx      = torch.cat([idx, idx_next])
    return tokenizer.decode(idx.tolist())
