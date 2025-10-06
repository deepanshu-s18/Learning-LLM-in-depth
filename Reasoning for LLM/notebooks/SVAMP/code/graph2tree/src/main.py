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
