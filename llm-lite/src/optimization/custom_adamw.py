"""
src/optimization/custom_adamw.py
Notebook 03: Adam Optimizer Demystified.
From-scratch implementation of AdamW with decoupled weight decay, exponentially weighted moving
averages (1st & 2nd moments), and mathematical bias corrections.
"""

import math
import torch
from typing import Dict, Any

class CustomAdamW(torch.optim.Optimizer):
    """
    Custom AdamW Optimizer implemented from first mathematical principles.
    
    Update Equations:
    1. First Moment (Momentum):
       m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
    2. Second Moment (RMSProp adaptive scale):
       v_t = beta_2 * v_{t-1} + (1 - beta_2) * (g_t ** 2)
    3. Bias Corrections (correcting initial zero bias):
       m_hat_t = m_t / (1 - beta_1 ** t)
       v_hat_t = v_t / (1 - beta_2 ** t)
    4. Decoupled Weight Decay + Step Update:
       theta_t = theta_{t-1} * (1 - lr * weight_decay) - lr * m_hat_t / (sqrt(v_hat_t) + eps)
    """
    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # 1. Decoupled Weight Decay: theta = theta * (1 - lr * weight_decay)
                if weight_decay != 0:
