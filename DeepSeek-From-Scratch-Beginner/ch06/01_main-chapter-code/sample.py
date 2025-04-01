# sample.py
# This script loads a trained MiniDeepSeek model and generates text from a
# user-provided prompt.
# UPDATED for PyTorch 2.6+ security features.

import os
import json
import torch
import tiktoken
from model import MiniDeepSeek, ModelArgs
# ## NEW ##: Import the serialization module to handle the new security feature
from torch import serialization

# --- Configuration ---
out_dir = 'out'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device.split(':')[0], dtype=pt_dtype) if 'cuda' in device else torch.no_grad()

