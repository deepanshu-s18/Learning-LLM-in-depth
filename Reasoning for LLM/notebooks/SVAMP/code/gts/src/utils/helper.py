import logging
import pdb
import torch
from glob import glob
from torch.autograd import Variable
import numpy as np
import os
import sys
# from src.utils.bleu import compute_bleu
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from src.utils.pre_data import *

def gpu_init_pytorch(gpu_num):
	'''
		Initialize GPU
	'''
	torch.cuda.set_device(int(gpu_num))
	device = torch.device("cuda:{}".format(
		gpu_num) if torch.cuda.is_available() else "cpu")
	return device

def create_save_directories(path):
	if not os.path.exists(path):
		os.makedirs(path)

def stack_to_string(stack):
	op = ""
	for i in stack:
		if op == "":
			op = op + i
		else:
			op = op + ' ' + i
	return op

def index_batch_to_words(input_batch, input_length, lang):
	'''
		Args:
			input_batch: List of BS x Max_len
			input_length: List of BS
		Return:
			contextual_input: List of BS
	'''
	contextual_input = []
	for i in range(len(input_batch)):
		contextual_input.append(stack_to_string(sentence_from_indexes(lang, input_batch[i][:input_length[i]])))
