import torch
from torch.nn import functional as F

from architecture.tokenizer import get_tokenizer




context_len=8192
tokenizer= get_tokenizer()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.tolist())



def generate_text(model, prompt, max_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt using trained model."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Tokenize input
    
    idx = text_to_token_ids(prompt,tokenizer).to(device)
    # Generate
    for _ in range(max_tokens):
        idx_cond = idx[-context_len:]
        with torch.inference_mode():
            logits= model(idx_cond)
        logits = logits[-1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[[-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=0)

    
    
       
    
    # Decode and return
    result = token_ids_to_text(idx,tokenizer)
    return result

