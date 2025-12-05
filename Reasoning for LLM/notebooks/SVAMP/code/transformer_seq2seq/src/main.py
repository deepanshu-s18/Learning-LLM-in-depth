# python -m src.main -no-debug -mode train -gpu 1 -dropout 0.1 -heads 4 -encoder_layers 1 -decoder_layers 1 -d_model 768 -d_ff 256 -lr 0.0001 -emb_lr 1e-5 -batch_size 32 -epochs 70 -embedding roberta -emb_name roberta-base -mawps_vocab -dataset mawps_fold0 -run_name mawps_try1
import os
import sys
import math
import logging
import pdb
import random
import numpy as np
from attrdict import AttrDict
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import OrderedDict
try:
	import cPickle as pickle
except ImportError:
	import pickle

from src.args import build_parser
from src.utils.helper import *
from src.utils.logger import get_logger, print_log, store_results, store_val_results
from src.dataloader import TextDataset
from src.model import build_model, train_model, run_validation

global log_folder
global model_folder
global result_folder
global data_path
global board_path

log_folder = 'logs'
model_folder = 'models'
outputs_folder = 'outputs'
result_folder = './out/'
data_path = './data/'
board_path = './runs/'

def load_data(config, logger):
	'''
		Loads the data from the datapath in torch dataset form

		Args:
			config (dict) : configuration/args
			logger (logger) : logger object for logging

		Returns:
			dataloader(s) 
	'''
	
	if config.mode == 'train':
		logger.debug('Loading Training Data...')

		'''Load Datasets'''
		train_set = TextDataset(data_path=data_path, dataset=config.dataset,
								datatype='train', max_length=config.max_length, is_debug=config.debug, is_train=True)
		val_set = TextDataset(data_path=data_path, dataset=config.dataset,
							  datatype='dev', max_length=config.max_length, is_debug=config.debug, grade_info=config.grade_disp, 
							  type_info=config.type_disp, challenge_info=config.challenge_disp)

		'''In case of sort by length, write a different case with shuffle=False '''
		train_dataloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=5)
		val_dataloader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True, num_workers=5)

		train_size = len(train_dataloader) * config.batch_size
		val_size = len(val_dataloader)* config.batch_size
		
		msg = 'Training and Validation Data Loaded:\nTrain Size: {}\nVal Size: {}'.format(train_size, val_size)
		logger.info(msg)

		return train_dataloader, val_dataloader

	elif config.mode == 'test':
		logger.debug('Loading Test Data...')

		test_set = TextDataset(data_path=data_path, dataset=config.dataset,
							   datatype='test', max_length=config.max_length, is_debug=config.debug)
		test_dataloader = DataLoader(
			test_set, batch_size=config.batch_size, shuffle=True, num_workers=5)

		logger.info('Test Data Loaded...')
		return test_dataloader

	else:
		logger.critical('Invalid Mode Specified')
		raise Exception('{} is not a valid mode'.format(config.mode))


def main():
