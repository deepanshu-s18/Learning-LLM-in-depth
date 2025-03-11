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
                    p.mul_(1.0 - lr * weight_decay)

                # 2. Update 1st and 2nd moments
                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 3. Bias corrections
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # 4. Parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def verify_adamw_against_pytorch() -> Dict[str, Any]:
    """
    Verifies that CustomAdamW matches PyTorch's official torch.optim.AdamW
    to high floating-point precision across multiple steps.
    """
    torch.manual_seed(42)
    p_custom = torch.nn.Parameter(torch.randn(10, 10, dtype=torch.float64))
    p_torch = torch.nn.Parameter(p_custom.clone().detach())

    opt_custom = CustomAdamW([p_custom], lr=1e-2, weight_decay=0.01)
    opt_torch = torch.optim.AdamW([p_torch], lr=1e-2, weight_decay=0.01)

    max_diff = 0.0
    for _ in range(5):
        # Generate dummy gradients
        grad = torch.randn_like(p_custom)
        p_custom.grad = grad.clone()
        p_torch.grad = grad.clone()

        opt_custom.step()
        opt_torch.step()

        diff = torch.max(torch.abs(p_custom - p_torch)).item()
        max_diff = max(max_diff, diff)

    return {
        "max_parameter_difference": max_diff,
        "parity_verified": max_diff < 1e-10
    }
