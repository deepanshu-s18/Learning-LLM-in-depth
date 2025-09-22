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
	'''
	torch.cuda.set_device(int(gpu_num))
	device = torch.device("cuda:{}".format(
		gpu_num) if torch.cuda.is_available() else "cpu")
	return device

def create_save_directories(path):
	if not os.path.exists(path):
		os.makedirs(path)

def save_checkpoint(state, epoch, logger, model_path, ckpt):
	'''
		Saves the model state along with epoch number. The name format is important for 
		the load functions. Don't mess with it.

		Args:
			model state
			epoch number
			logger variable
			directory to save models
			checkpoint name
	'''
	ckpt_path = os.path.join(model_path, '{}.pt'.format(ckpt))
	logger.info('Saving Checkpoint at : {}'.format(ckpt_path))
	torch.save(state, ckpt_path)

def get_latest_checkpoint(model_path, logger):
