from training.data_loader import train_loader,val_loader
from architecture.gptoss import Transformer,ModelConfig
import torch
from inference import generate_text
device= "cuda:0"
