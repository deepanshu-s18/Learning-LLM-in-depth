"""
src/compression/quantizer.py
Notebook 11: LLM Quantization (FP32 to INT8 & INT4).
Implements affine linear quantization, scale & zero-point calibration,
4-bit nibble packing/unpacking, and quantization error evaluation (MSE, SNR, memory reduction).
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

def calculate_scale_zero_point(
    min_val: float,
    max_val: float,
    qmin: int,
    qmax: int
) -> Tuple[float, int]:
    """
    Computes scale (S) and zero-point (Z) affine parameters.
    S = (max - min) / (qmax - qmin)
    Z = round(-min / S) + qmin
    """
    if max_val == min_val:
        return 1.0, 0
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = int(round(-min_val / scale)) + qmin
    zero_point = max(qmin, min(qmax, zero_point))
    return scale, zero_point


def quantize_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
    """
    Quantizes FP32 tensor to INT8 format (range [-128, 127]).
    """
    qmin, qmax = -128, 127
    min_val = float(tensor.min().item())
    max_val = float(tensor.max().item())
    scale, zero_point = calculate_scale_zero_point(min_val, max_val, qmin, qmax)

    # q = clamp(round(x / S + Z), qmin, qmax)
    q_tensor = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax).to(torch.int8)
    return q_tensor, scale, zero_point


def dequantize_int8(q_tensor: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """
    Dequantizes INT8 tensor back to FP32.
    x_hat = S * (q - Z)
    """
    return scale * (q_tensor.to(torch.float32) - zero_point)


def quantize_int4(tensor: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
    """
    Quantizes FP32 tensor to INT4 range [-8, 7] and packs two 4-bit values into one uint8 byte.
    """
    qmin, qmax = -8, 7
    min_val = float(tensor.min().item())
    max_val = float(tensor.max().item())
    scale, zero_point = calculate_scale_zero_point(min_val, max_val, qmin, qmax)

    q = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax).to(torch.int32)
    # Shift to unsigned [0, 15] for nibble packing
    q_unsigned = (q - qmin).to(torch.uint8)

    # Pack pairs of 4-bit numbers: (high_nibble << 4) | low_nibble
    flat = q_unsigned.flatten()
