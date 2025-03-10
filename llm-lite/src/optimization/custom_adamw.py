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
