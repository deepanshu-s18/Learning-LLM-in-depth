import torch
import gc
from torch.utils.data import Dataset, DataLoader
from architecture.tokenizer import get_tokenizer
from datasets import load_dataset

batch_size  = 4
context_len = 2048

dataset    = load_dataset("roneneldan/TinyStories")
train_text = " ".join(ex["text"] for ex in dataset["train"])
val_text   = " ".join(ex["text"] for ex in dataset["validation"])

tokenizer    = get_tokenizer()
train_tokens = tokenizer.encode(train_text)
val_tokens   = tokenizer.encode(val_text)


class TextDataset(Dataset):
    def __init__(self, tokens, max_len=4000, stride=4000):
