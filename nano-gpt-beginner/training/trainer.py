import os
import time
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from inference import generate_text

def calc_loss_batch(x, y, model, device):
    x, y = x.to(device), y.to(device)
    logits = model(x)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())

def calc_loss_loader(loader, model, device, num_batches=None):
    total = 0.0
    n     = len(loader) if num_batches is None else min(num_batches, len(loader))
    for i, (x, y) in enumerate(loader):
        if i >= n:
            break
