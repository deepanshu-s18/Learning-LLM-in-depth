from training.data_loader import train_loader,val_loader
from architecture.gptoss import Transformer,ModelConfig
import torch
from inference import generate_text
device= "cuda:0"
context="Once upon a day"
model= Transformer(ModelConfig(
    num_attention_heads=8
    ,num_key_value_heads=4,
    num_experts=4,
    experts_per_token=1,
    num_hidden_layers=12,
    hidden_size =1024,
    intermediate_size = 1024
    
