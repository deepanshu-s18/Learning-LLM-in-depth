import torch
import gc
from torch.utils.data import Dataset, DataLoader
from architecture.tokenizer import get_tokenizer
from datasets import load_dataset

batch_size  = 4
context_len = 2048

dataset    = load_dataset("roneneldan/TinyStories")
