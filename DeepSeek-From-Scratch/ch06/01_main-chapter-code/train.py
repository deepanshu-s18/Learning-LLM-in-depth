# train.py
# This script trains the MiniDeepSeek model on the tokenized TinyStories dataset.
# It is designed to run on a single GPU.

import os
import json
import time
from contextlib import nullcontext
import math # Added for learning rate scheduler

import numpy as np
import torch
from torch.nn import functional as F

# Import the model definition from model.py
from model import MiniDeepSeek, ModelArgs

# --- Configuration ---
# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
