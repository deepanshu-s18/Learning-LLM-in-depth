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

			train_ls, dev_ls = load_raw_data(data_path, config.dataset)
			pairs_trained, pairs_tested, generate_nums, copy_nums = transfer_num(train_ls, dev_ls, config.challenge_disp)

			logger.debug('Data Loaded...')
			logger.debug('Number of Training Examples: {}'.format(len(pairs_trained)))
			logger.debug('Number of Testing Examples: {}'.format(len(pairs_tested)))
			logger.debug('Extra Numbers: {}'.format(generate_nums))
			logger.debug('Maximum Number of Numbers: {}'.format(copy_nums))

			# pairs: ([list of words in question], [list of infix Equation tokens incl brackets and N0, N1], [list of numbers], [list of indexes of numbers])
			# generate_nums: Unmentioned numbers used in eqns in atleast 5 examples ['1', '3.14']
			# copy_nums: Maximum number of numbers in a single sentence: 15

			# pairs: ([list of words in question], [list of prefix Equation tokens w/ metasymbols as N0, N1], [list of numbers], [list of indexes of numbers])

			logger.info('Creating Vocab...')
			input_lang = None
			output_lang = None

			input_lang, output_lang, train_pairs, test_pairs = prepare_data(config, logger, pairs_trained, pairs_tested, config.trim_threshold, generate_nums, copy_nums, input_lang, output_lang, tree=True)

