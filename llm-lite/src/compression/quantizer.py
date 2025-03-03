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
