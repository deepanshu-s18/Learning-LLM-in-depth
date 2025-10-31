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

	return contextual_input

def sort_by_len(seqs, input_len, device=None, dim=1):
	orig_idx = list(range(seqs.size(dim)))
	# pdb.set_trace()

	# Index by which sorting needs to be done
	sorted_idx = sorted(orig_idx, key=lambda k: input_len[k], reverse=True)
	sorted_idx= torch.LongTensor(sorted_idx)
	if device:
		sorted_idx = sorted_idx.to(device)

	sorted_seqs = seqs.index_select(1, sorted_idx)
	sorted_lens=  [input_len[i] for i in sorted_idx]

	# For restoring original order
	orig_idx = sorted(orig_idx, key=lambda k: sorted_idx[k])
	orig_idx = torch.LongTensor(orig_idx)
	if device:
		orig_idx = orig_idx.to(device)
		# sorted_lens = torch.LongTensor(sorted_lens).to(device)
	return sorted_seqs, sorted_lens, orig_idx

def save_checkpoint(state, epoch, logger, model_path, ckpt):
