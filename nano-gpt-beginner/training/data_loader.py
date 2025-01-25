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
        self.inputs  = []
        self.targets = []
        for i in range(0, len(tokens) - max_len, stride):
            self.inputs.append(torch.tensor(tokens[i: i + max_len]))
            self.targets.append(torch.tensor(tokens[i + 1: i + max_len + 1]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


train_dataset = TextDataset(train_tokens, context_len, context_len)
val_dataset   = TextDataset(val_tokens,   context_len, context_len)

train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

del dataset, train_text, val_text
gc.collect()
