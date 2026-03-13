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
            
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
            
        return self.output(x)

class Diffusion(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.model = SimpleUNet(config).to(config.device)
        
        self.beta = torch.linspace(config.beta_start, config.beta_end, config.timesteps).to(config.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        ε = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * ε, ε

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.config.timesteps, size=(n,), device=self.config.device)

    def forward(self, x):
        t = self.sample_timesteps(x.shape[0])
        x_t, noise = self.noise_images(x, t)
        predicted_noise = self.model(x_t, t)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def sample(self, n_samples):
        self.model.eval()
        x = torch.randn((n_samples, self.config.in_channels, self.config.image_size, self.config.image_size)).to(self.config.device)
        
        for i in reversed(range(1, self.config.timesteps)):
            t = (torch.ones(n_samples) * i).long().to(self.config.device)
            predicted_noise = self.model(x, t)
            
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
        self.model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return x
```

**Our Promise:** In the next 40 minutes, this file will be completely demystified. You will understand not just what each line does, but *why* it's there.

**Key Formulas & Concepts You Will Master:**

*   The Forward Process "Shortcut" Formula:
    $$ \mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon} $$
*   The Reverse Process (DDPM Sampling) Formula:
    $$ \mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z} $$
*   The Training Objective: A simple Mean Squared Error, `MSE($\epsilon$, $\epsilon_\theta$)`.
*   The `SimpleUNet` architecture, `SinusoidalPositionEmbeddings`, and how to condition a model on time.


## **Chapter 1: Our Blueprint - The `DiffusionConfig`**

Every complex project, from a skyscraper to a neural network, starts with a blueprint. This blueprint defines the key parameters and dimensions that guide the entire construction. In our `diffusion_min.py` file, this role is played by the `DiffusionConfig` class.

Let's look at the code we are about to build.

```python
# diffusion_min.py (lines 7-17)
from dataclasses import dataclass
import torch

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
```

This is a Python `@dataclass`, which is just a clean and simple way to group variables together. It's a container that holds all the knobs we can turn to change the size, speed, and behavior of our diffusion model.

Before we write a single line of logic, we define these core parameters. Let's go through them one by one.

| Parameter | What it Controls | Intuition | `diffusion_min.py` Value |
| :--- | :--- | :--- | :--- |
| `image_size` | The height and width of the images we will generate. | The size of our digital canvas. | `32` (e.g., 32x32 pixels) |
| `in_channels`| The number of color channels in our input images. | Is it a grayscale image (1) or a color image (3 for RGB)? | `3` |
| `base_channels`| The initial number of channels in the first layer of our U-Net. | Controls the **width** and overall size of our neural network. A larger number means a more powerful (and slower) model. | `64` |
| `time_emb_dim`| The dimensionality of the vector that will represent the timestep `t`. | We need to tell our network whether we're at the beginning, middle, or end of the denoising process. This defines the "size" of that time signal. | `256` |
| `timesteps` | The total number of noising steps in the forward process (`T`). | How many steps of destruction will we apply? This also defines the number of steps our model will take to generate an image. | `1000` |
| `beta_start` | The noise variance ($\beta$) for the very first timestep (`t=1`). | How much noise do we add at the very beginning? | `0.0001` (a tiny amount) |
| `beta_end` | The noise variance ($\beta$) for the final timestep (`t=T`). | How much noise do we add at the very end? | `0.02` (a larger amount) |
| `device` | Standard PyTorch boilerplate for device management. | Do we run on the GPU (`cuda`) or CPU? | `"cuda" if available` |

These parameters are the foundation of our entire model. The `timesteps`, `beta_start`, and `beta_end` are especially important, as they define the "noise schedule" which is the heart of the forward process.

With our blueprint defined, we are now ready to build the first major component of our model: the fixed, mathematical process of destroying an image with noise.

## **Chapter 2: The Forward Process: Math & Intuition**

The forward process destroys an image by adding noise, one step at a time. Here's the formula for a single step:

$$ \mathbf{x}_t = \sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \boldsymbol{\epsilon} $$

| Term | What it is | Intuition |
|:-----|:-----------|:----------|
| $x_t$ | The noisy image at step $t$ | Our output—slightly noisier than before |
| $x_{t-1}$ | The image from the previous step | What we're corrupting |
| $\epsilon$ | Fresh Gaussian noise $\sim \mathcal{N}(0, 1)$ | Pure random static |
| $\beta_t$ | Noise variance at step $t$ | How much noise to add (small, e.g. 0.0001 to 0.02) |
| $\alpha_t = 1 - \beta_t$ | Signal retention rate | How much of the previous image to keep (close to 1) |
| $\sqrt{\alpha_t}$ | Scale factor for image | We scale down the image slightly... |
| $\sqrt{\beta_t}$ | Scale factor for noise | ...and add a small amount of noise |

**Why square roots?** We're working with *variances*, not standard deviations. When you scale a random variable by $c$, its variance scales by $c^2$. So to add noise with variance $\beta_t$, we multiply by $\sqrt{\beta_t}$. The square roots ensure the total variance stays controlled: $(\sqrt{\alpha_t})^2 + (\sqrt{\beta_t})^2 = \alpha_t + \beta_t = 1$.

#### The Variance Schedule ($\beta_t$)

The key to the forward process is the **variance schedule**, denoted $\beta_t$ (beta). This schedule dictates exactly how much noise we add at each timestep $t$. In the original DDPM paper, this is a simple linear schedule:

*   At $t=1$, we add a tiny amount of noise: $\beta_1 = 0.0001$
*   At $t=1000$, we add more noise: $\beta_{1000} = 0.02$
*   The values for $\beta_2$, $\beta_3$, etc. are evenly spaced between these endpoints.

This means we start by adding just a whisper of noise, and gradually add more at each subsequent step.

From $\beta_t$, we derive $\alpha_t = 1 - \beta_t$. If $\beta_t$ is the noise rate, then $\alpha_t$ is the **signal rate**—how much of the previous image we keep. Since $\beta_t$ is always small, $\alpha_t$ is always close to 1 (e.g., 0.9999).

#### The Problem: This is Slow

To get a noisy image $x_t$ from the original $x_0$, we'd have to apply the formula $t$ times in sequence:

$$x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \cdots \rightarrow x_t$$

For $t = 500$, that's 500 sequential operations. This would make training painfully slow.

#### The Shortcut

Let's derive a formula that jumps directly from $x_0$ to $x_t$. Start with the one-step formula and expand it:

$$\mathbf{x}_1 = \sqrt{\alpha_1} \mathbf{x}_0 + \sqrt{\beta_1} \boldsymbol{\epsilon}_1$$

$$\mathbf{x}_2 = \sqrt{\alpha_2} \mathbf{x}_1 + \sqrt{\beta_2} \boldsymbol{\epsilon}_2$$

Substitute $x_1$ into the equation for $x_2$:

$$\mathbf{x}_2 = \sqrt{\alpha_2} \left( \sqrt{\alpha_1} \mathbf{x}_0 + \sqrt{\beta_1} \boldsymbol{\epsilon}_1 \right) + \sqrt{\beta_2} \boldsymbol{\epsilon}_2$$

$$= \sqrt{\alpha_1 \alpha_2} \mathbf{x}_0 + \sqrt{\alpha_2 \beta_1} \boldsymbol{\epsilon}_1 + \sqrt{\beta_2} \boldsymbol{\epsilon}_2$$

Here's the key insight: when you add two independent Gaussian random variables, the result is also Gaussian, with variances that add. So the two noise terms combine:

$$\sqrt{\alpha_2 \beta_1} \boldsymbol{\epsilon}_1 + \sqrt{\beta_2} \boldsymbol{\epsilon}_2 \sim \mathcal{N}(0, \alpha_2 \beta_1 + \beta_2)$$

Since $\beta_1 = 1 - \alpha_1$, we can simplify:
$$\alpha_2 \beta_1 + \beta_2 = \alpha_2(1 - \alpha_1) + (1 - \alpha_2) = 1 - \alpha_1 \alpha_2$$

So we can write:
$$\mathbf{x}_2 = \sqrt{\alpha_1 \alpha_2} \mathbf{x}_0 + \sqrt{1 - \alpha_1 \alpha_2} \boldsymbol{\epsilon}$$

The pattern is clear. Continuing this for $t$ steps gives us:

$$ \mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon} $$

where $\bar{\alpha}_t = \alpha_1 \times \alpha_2 \times \cdots \times \alpha_t$ is the cumulative product.

This is the single most important equation for the forward process. Let's break it down:

*   $x_0$ is our original, clean image.
*   $\epsilon$ is a single sample of pure Gaussian noise.
*   $\bar{\alpha}_t$ (alpha-bar) is the cumulative product: $\bar{\alpha}_t = \alpha_1 \times \alpha_2 \times \cdots \times \alpha_t$

The term $\bar{\alpha}_t$ tells us how much of the original image signal remains at timestep $t$. As $t$ increases, $\bar{\alpha}_t$ decays from 1.0 toward 0.0—the original signal fades away and only noise remains.

The formula is just a weighted sum: we take $\sqrt{\bar{\alpha}_t}$ of the original image and add $\sqrt{1 - \bar{\alpha}_t}$ of pure noise. When $t$ is small, we keep most of the image. When $t$ is large, we keep almost none—just noise.

This shortcut is essential for efficient training. We can generate a training sample $(x_t, \epsilon)$ for any random $t$ on the fly, without any iteration. In the next chapter, we'll translate this formula directly into PyTorch code.

## **Chapter 3: The Forward Process: Code Implementation**

We need to implement this formula in PyTorch:

$$ \mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon} $$

Looking at this formula, we need three things:
1. $\bar{\alpha}_t$ — the cumulative signal rate, precomputed for all timesteps
2. $x_0$ — the clean image (input)
3. $\epsilon$ — fresh Gaussian noise (we'll sample this on the fly)

Since $\bar{\alpha}_t$ depends only on the timestep (not the image), we can precompute it once and reuse it. Our implementation has two pieces:
1.  The pre-computation of our noise schedules ($\beta_t$, $\alpha_t$, $\bar{\alpha}_t$).
2.  The "shortcut" function that generates a noisy image $x_t$ for any given $x_0$ and $t$.

#### Part 1: Pre-computing the Schedules in `Diffusion.__init__`

Recall that $\bar{\alpha}_t = \alpha_1 \times \alpha_2 \times \cdots \times \alpha_t$, where $\alpha_t = 1 - \beta_t$. We need to:
1. Create the $\beta_t$ schedule (linearly spaced values)
2. Compute $\alpha_t = 1 - \beta_t$
3. Compute $\bar{\alpha}_t$ as the cumulative product of $\alpha$

These are fixed constants, so we compute them once at initialization:

```python
# diffusion_min.py (lines 118-124)
class Diffusion(nn.Module):
    def __init__(self, config: DiffusionConfig):
