from training.data_loader import train_loader, val_loader
from architecture.gptoss import Transformer, ModelConfig
from training.trainer import trainer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Transformer(ModelConfig(
    num_attention_heads=8,
    num_key_value_heads=4,
    num_experts=4,
    experts_per_token=1,
    num_hidden_layers=12,
    hidden_size=1024,
    intermediate_size=1024,
), device)

print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
trainer(model, train_loader, val_loader, device)
