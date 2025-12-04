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
