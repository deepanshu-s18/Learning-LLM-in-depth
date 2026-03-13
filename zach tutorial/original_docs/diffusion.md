# **Title: Diffusion Models From Scratch in 40 Minutes**

> Diffusion Models Simply Explained

## **Introduction: The Promise**

You've heard of Stable Diffusion. You've seen DALL-E and Midjourney generate fascinating images from nothing but a text prompt. It feels like magic—an unknowable black box of artificial intelligence.

But it's not magic. The core engine behind these models is a Denoising Diffusion Probabilistic Model (DDPM), and its core idea is surprisingly simple.

1.  **Destroy:** First, we take a beautiful, clean image and systematically destroy it by adding step-by-step Gaussian noise until only static remains. This is a fixed, mathematical process called the **Forward Process**.
2.  **Teach:** Second, we train a neural network on a single, focused task: to look at a noisy image and predict the noise that was added to it. It learns to undo one small step of the destruction.
3.  **Create:** Finally, to create something new, we give the trained network a canvas of pure static and tell it: "Heal this." The network applies its knowledge step-by-step, progressively removing noise until a clean, original image emerges. This is the **Reverse Process**.

The entire secret is in the Python code below.

```python
# diffusion_min.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class DiffusionConfig:
    image_size: int = 32
    in_channels: int = 3
    base_channels: int = 64
    time_emb_dim: int = 256
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """Simple Conv -> GroupNorm -> GELU block"""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.GroupNorm(8, out_ch)
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.GELU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class SimpleUNet(nn.Module):
    """A minimal U-Net to predict noise"""
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        image_channels = config.in_channels
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = config.in_channels 
        time_emb_dim = config.time_emb_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU()
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) \
                                    for i in range(len(down_channels)-1)])
        
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) \
                                  for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
            
