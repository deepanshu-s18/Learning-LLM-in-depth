"""
src/foundations/pytorch_engine.py
Notebook 02: PyTorch Deep Dive & Production Training Harness.
Provides device abstraction (MPS/CUDA/CPU), autograd utilities, gradient clipping,
checkpoint serialization, and training tracking.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
