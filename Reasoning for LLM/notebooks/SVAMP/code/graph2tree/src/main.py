# coding: utf-8
import time
import torch.optim
from collections import OrderedDict
from attrdict import AttrDict
import pandas as pd
try:
	import cPickle as pickle
except ImportError:
	import pickle
import json
import pdb

from src.args import build_parser

from src.train_and_evaluate import *
from src.components.models import *
from src.components.contextual_embeddings import *
from src.utils.helper import *
from src.utils.logger import *
from src.utils.expressions_transfer import *

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

def read_json(path):
	with open(path,'r') as f:
		file = json.load(f)
	return file

USE_CUDA = True

def get_new_fold(data,pairs,group):
	new_fold = []
	for item,pair,g in zip(data, pairs, group):
		pair = list(pair)
		pair.append(g['group_num'])
		pair = tuple(pair)
		new_fold.append(pair)
	return new_fold

def change_num(num):
	new_num = []
	for item in num:
		if '/' in item:
			new_str = item.split(')')[0]
			new_str = new_str.split('(')[1]
			a = float(new_str.split('/')[0])
			b = float(new_str.split('/')[1])
			value = a/b
			new_num.append(value)
		elif '%' in item:
			value = float(item[0:-1])/100
			new_num.append(value)
		else:
			new_num.append(float(item))
	return new_num

def main():
	parser = build_parser()
	args = parser.parse_args()
	config = args

	if config.mode == 'train':
		is_train = True
	else:
		is_train = False

	''' Set seed for reproducibility'''
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	random.seed(config.seed)

	'''GPU initialization'''
	device = gpu_init_pytorch(config.gpu)

	if config.full_cv:
		global data_path
		data_name = config.dataset
		data_path = data_path + data_name + '/'
		config.val_result_path = os.path.join(result_folder, 'CV_results_{}.json'.format(data_name))
		fold_acc_score = 0.0
		folds_scores = []
		best_acc = []
		for z in range(5):
			run_name = config.run_name + '_fold' + str(z)
			config.dataset = 'fold' + str(z)
			config.log_path = os.path.join(log_folder, run_name)
			config.model_path = os.path.join(model_folder, run_name)
			config.board_path = os.path.join(board_path, run_name)
			config.outputs_path = os.path.join(outputs_folder, run_name)

			vocab1_path = os.path.join(config.model_path, 'vocab1.p')
			vocab2_path = os.path.join(config.model_path, 'vocab2.p')
			config_file = os.path.join(config.model_path, 'config.p')
			log_file = os.path.join(config.log_path, 'log.txt')

			if config.results:
