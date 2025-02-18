import torch
from torch.nn import functional as F

from architecture.tokenizer import get_tokenizer




context_len=8192
tokenizer= get_tokenizer()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
