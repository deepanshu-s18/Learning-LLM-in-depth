
import torch,gc
from torch.utils.data import Dataset,DataLoader
from architecture.tokenizer import get_tokenizer
from datasets import load_dataset
from tqdm.notebook import tqdm
batch_size=5
context_len=4000

dataset = load_dataset("roneneldan/TinyStories")
train_text = " ".join([ex["text"] for ex in dataset['train']])
val_text = " ".join([ex["text"] for ex in dataset['validation']])

tokenizer = get_tokenizer()
print("tokenizing...")
train_tokens = tokenizer.encode(train_text)
val_tokens = tokenizer.encode(val_text)
print("tokenized")
class TextDataset(Dataset):
    def __init__(self, tokens, max_length=8192, stride=8192):
        self.input_ids = []
        self.target_ids = []
        for i in tqdm(range(0, len(tokens) - max_length, stride)):
            input_chunk = tokens[i:i + max_length]
            target_chunk = tokens[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

train_dataset = TextDataset(train_tokens, max_length=context_len, stride=context_len)
val_dataset = TextDataset(val_tokens, max_length=context_len, stride=context_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

del dataset, train_text, val_text
gc.collect()