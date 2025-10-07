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
				config.result_path = os.path.join(result_folder, 'val_results_{}.json'.format(config.dataset))

			create_save_directories(config.log_path)
			create_save_directories(config.model_path)
			create_save_directories(config.outputs_path)

			logger = get_logger(run_name, log_file, logging.DEBUG)

			logger.info('Experiment Name: {}'.format(config.run_name))
			logger.debug('Created Relevant Directories')

			logger.info('Loading Data...')

			train_ls, dev_ls = load_raw_data(data_path, config.dataset, is_train)

			pairs_trained, pairs_tested, generate_nums, copy_nums = transfer_num(train_ls, dev_ls, config.challenge_disp)

			logger.debug('Data Loaded...')
			logger.debug('Number of Training Examples: {}'.format(len(pairs_trained)))
			logger.debug('Number of Testing Examples: {}'.format(len(pairs_tested)))
			logger.debug('Extra Numbers: {}'.format(generate_nums))
			logger.debug('Maximum Number of Numbers: {}'.format(copy_nums))

			logger.info('Creating Vocab...')
			input_lang = None
			output_lang = None

			input_lang, output_lang, train_pairs, test_pairs = prepare_data(config, logger, pairs_trained, pairs_tested, config.trim_threshold, generate_nums, copy_nums, input_lang, output_lang, tree=True)

			checkpoint = get_latest_checkpoint(config.model_path, logger)

			with open(vocab1_path, 'wb') as f:
				pickle.dump(input_lang, f, protocol=pickle.HIGHEST_PROTOCOL)
			with open(vocab2_path, 'wb') as f:
				pickle.dump(output_lang, f, protocol=pickle.HIGHEST_PROTOCOL)

			logger.debug('Vocab saved at {}'.format(vocab1_path))

			generate_num_ids = []
			for num in generate_nums:
				generate_num_ids.append(output_lang.word2index[num])

			config.len_generate_nums = len(generate_nums)
			config.copy_nums = copy_nums

			with open(config_file, 'wb') as f:
				pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

			logger.debug('Config File Saved')

			logger.info('Initializing Models...')

			# Initialize models
			embedding = None
			if config.embedding == 'bert':
				embedding = BertEncoder(config.emb_name, device, config.freeze_emb)
			elif config.embedding == 'roberta':
				embedding = RobertaEncoder(config.emb_name, device, config.freeze_emb)
			else:
				embedding = Embedding(config, input_lang, input_size=input_lang.n_words, embedding_size=config.embedding_size, dropout=config.dropout)

			encoder = EncoderSeq(cell_type=config.cell_type, embedding_size=config.embedding_size, hidden_size=config.hidden_size, n_layers=config.depth, dropout=config.dropout)
			predict = Prediction(hidden_size=config.hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums), input_size=len(generate_nums), dropout=config.dropout)
			generate = GenerateNode(hidden_size=config.hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums), embedding_size=config.embedding_size, dropout=config.dropout)
			merge = Merge(hidden_size=config.hidden_size, embedding_size=config.embedding_size, dropout=config.dropout)
			# the embedding layer is  only for generated number embeddings, operators, and paddings

			logger.debug('Models Initialized')
			logger.info('Initializing Optimizers...')

			embedding_optimizer = torch.optim.Adam(embedding.parameters(), lr=config.emb_lr, weight_decay=config.weight_decay)
			encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay)
			predict_optimizer = torch.optim.Adam(predict.parameters(), lr=config.lr, weight_decay=config.weight_decay)
			generate_optimizer = torch.optim.Adam(generate.parameters(), lr=config.lr, weight_decay=config.weight_decay)
			merge_optimizer = torch.optim.Adam(merge.parameters(), lr=config.lr, weight_decay=config.weight_decay)

			logger.debug('Optimizers Initialized')
			logger.info('Initializing Schedulers...')

			embedding_scheduler = torch.optim.lr_scheduler.StepLR(embedding_optimizer, step_size=20, gamma=0.5)
			encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
			predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
			generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
			merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)

			logger.debug('Schedulers Initialized')

