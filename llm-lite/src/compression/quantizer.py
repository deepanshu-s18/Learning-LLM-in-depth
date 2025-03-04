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
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
    
    high = flat[0::2] << 4
    low = flat[1::2] & 0x0F
    packed = high | low

    return packed, scale, zero_point


def dequantize_int4(packed: torch.Tensor, original_shape: torch.Size, scale: float, zero_point: int) -> torch.Tensor:
    """
    Unpacks 4-bit nibbles from uint8 bytes and dequantizes back to FP32.
    """
    qmin = -8
    # Unpack
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    unpacked = torch.stack([high, low], dim=-1).flatten()
    
    # Restore signed range
    q = unpacked[: original_shape.numel()].view(original_shape).to(torch.int32) + qmin
    return scale * (q.to(torch.float32) - zero_point)


def benchmark_model_quantization(model: nn.Module) -> Dict[str, Any]:
    """
    Quantizes all linear weights of the model to INT8 and INT4,
    evaluating memory reduction and reconstruction Signal-to-Noise Ratio (SNR).
    """
    total_fp32_bytes = 0
    total_int8_bytes = 0
    total_int4_bytes = 0
    mse_int8_list = []
    mse_int4_list = []

    for name, p in model.named_parameters():
        if "weight" in name and p.dim() >= 2:
            data = p.data
            numel = data.numel()
            total_fp32_bytes += numel * 4 # 4 bytes for float32
            total_int8_bytes += numel * 1 # 1 byte for int8
            total_int4_bytes += math.ceil(numel / 2) # 0.5 byte for int4

            # INT8 evaluation
            q8, s8, z8 = quantize_int8(data)
            deq8 = dequantize_int8(q8, s8, z8)
            mse8 = torch.mean((data - deq8) ** 2).item()
            mse_int8_list.append(mse8)

            # INT4 evaluation
            q4, s4, z4 = quantize_int4(data)
            deq4 = dequantize_int4(q4, data.shape, s4, z4)
            mse4 = torch.mean((data - deq4) ** 2).item()
            mse_int4_list.append(mse4)

    avg_mse_int8 = sum(mse_int8_list) / max(len(mse_int8_list), 1)
    avg_mse_int4 = sum(mse_int4_list) / max(len(mse_int4_list), 1)

    return {
        "fp32_memory_kb": total_fp32_bytes / 1024.0,
        "int8_memory_kb": total_int8_bytes / 1024.0,
        "int4_memory_kb": total_int4_bytes / 1024.0,
        "int8_compression_ratio": f"{(total_fp32_bytes / max(total_int8_bytes, 1)):.1f}x",
        "int4_compression_ratio": f"{(total_fp32_bytes / max(total_int4_bytes, 1)):.1f}x",
        "int8_average_mse": avg_mse_int8,
        "int4_average_mse": avg_mse_int4
    }
