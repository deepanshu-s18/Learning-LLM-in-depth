import math
import torch
import torch.optim
import torch.nn.functional as f
import time
import pdb

from src.components.models import *
from src.components.masked_cross_entropy import *
from src.utils.pre_data import *
from src.utils.expressions_transfer import *
from src.utils.helper import *

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


class Beam:  # the class save the beam node
	def __init__(self, score, input_var, hidden, all_output):
		self.score = score
		self.input_var = input_var
		self.hidden = hidden
		self.all_output = all_output


def time_since(s):  # compute time
	m = math.floor(s / 60)
	s -= m * 60
	h = math.floor(m / 60)
	m -= h * 60
	return '%dh %dm %ds' % (h, m, s)
