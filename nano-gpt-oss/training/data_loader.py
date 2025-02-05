
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
