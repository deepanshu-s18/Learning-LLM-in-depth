import logging
import pdb
import torch
from glob import glob
from torch.autograd import Variable
import numpy as np
import os
import sys
from src.utils.bleu import compute_bleu
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def gpu_init_pytorch(gpu_num):
	'''
		Initialize GPU

		Args:
			gpu_num (int): Which GPU to use
		Returns:
			device (torch.device): GPU device
