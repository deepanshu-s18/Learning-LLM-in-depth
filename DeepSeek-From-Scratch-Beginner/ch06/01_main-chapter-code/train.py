import os, json, time, math
import numpy as np
import torch
from model import MiniDeepSeek, ModelArgs

device     = "cuda" if torch.cuda.is_available() else "cpu"
data_dir   = "data/tinystories_tokenized"
out_dir    = "out"
max_iters  = 5000
batch_size = 32
block_size = 128
lr         = 3e-4
