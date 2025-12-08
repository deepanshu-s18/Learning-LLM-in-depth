import os
import sys
import math
import logging
import pdb
import random
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW
# from pytorch_pretrained_bert.optimization import BertAdam
# from tensorboardX import SummaryWriter
from gensim import models
from src.components.contextual_embeddings import BertEncoder, RobertaEncoder
from src.utils.sentence_processing import *
from src.utils.logger import print_log, store_results
from src.utils.helper import save_checkpoint, bleu_scorer
from src.utils.evaluate import cal_score, stack_to_string, get_infix_eq

from collections import OrderedDict

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.scale = nn.Parameter(torch.ones(1)) # nn.Parameter causes the tensor to appear in the model.parameters()

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # max_len x 1
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # torch.arange(0, d_model, 2) gives 2i
		pe[:, 0::2] = torch.sin(position * div_term) # all alternate columns 0 onwards
		pe[:, 1::2] = torch.cos(position * div_term) # all alternate columns 1 onwards
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		'''
