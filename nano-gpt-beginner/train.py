from training.data_loader import train_loader, val_loader
from architecture.gptoss import Transformer, ModelConfig
from training.trainer import trainer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Transformer(ModelConfig(
    num_attention_heads=8,
