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

			logger.info('Loading Models on GPU {}...'.format(config.gpu))

			# Move models to GPU
			if USE_CUDA:
				embedding.to(device)
				encoder.to(device)
				predict.to(device)
				generate.to(device)
				merge.to(device)

			logger.debug('Models loaded on GPU {}'.format(config.gpu))

			max_value_corr = 0
			len_total_eval = 0
			max_val_acc = 0.0
			max_train_acc = 0.0
			eq_acc = 0.0
			best_epoch = -1
			min_train_loss = float('inf')

			logger.info('Starting Training Procedure')

			for epoch in range(config.epochs):
				loss_total = 0
				input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, graph_batches, group_batches = prepare_train_batch(train_pairs, config.batch_size)
				
				od = OrderedDict()
				od['Epoch'] = epoch + 1
				print_log(logger, od)

				start = time.time()
				for idx in range(len(input_lengths)):
					loss = train_tree(
						config, input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
						num_stack_batches[idx], num_size_batches[idx], num_value_batches[idx], group_batches[idx], generate_num_ids, embedding, encoder, predict, generate, merge,
						embedding_optimizer, encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, input_lang, output_lang, 
						num_pos_batches[idx], graph_batches[idx])
					loss_total += loss
					print("Completed {} / {}...".format(idx, len(input_lengths)), end = '\r', flush = True)

				embedding_scheduler.step()
				encoder_scheduler.step()
				predict_scheduler.step()
				generate_scheduler.step()
				merge_scheduler.step()

				logger.debug('Training for epoch {} completed...\nTime Taken: {}'.format(epoch, time_since(time.time() - start)))
				
				if loss_total / len(input_lengths) < min_train_loss:
					min_train_loss = loss_total / len(input_lengths)

				train_value_ac = 0
				train_equation_ac = 0
				train_eval_total = 1
				if config.show_train_acc:
					train_eval_total = 0
					logger.info('Computing Train Accuracy')
					start = time.time()
					with torch.no_grad():
						for train_batch in train_pairs:
							batch_graph = get_single_example_graph(train_batch[0], train_batch[1], train_batch[7], train_batch[4], train_batch[5])
							train_res = evaluate_tree(config, train_batch[0], train_batch[1], generate_num_ids, embedding, encoder, predict, generate,
													 merge, input_lang, output_lang, train_batch[4], train_batch[5], batch_graph, test_batch[7], beam_size=config.beam_size)
							train_val_ac, train_equ_ac, _, _ = compute_prefix_tree_result(train_res, train_batch[2], output_lang, train_batch[4], train_batch[6])

							if train_val_ac:
								train_value_ac += 1
							if train_equ_ac:
								train_equation_ac += 1
							train_eval_total += 1

					logger.debug('Train Accuracy Computed...\nTime Taken: {}'.format(time_since(time.time() - start)))

				logger.info('Starting Validation')

				value_ac = 0
				equation_ac = 0
				eval_total = 0
				start = time.time()

				with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
					f_out.write('---------------------------------------\n')
					f_out.write('Epoch: ' + str(epoch) + '\n')
					f_out.write('---------------------------------------\n')
					f_out.close()

				ex_num = 0
				for test_batch in test_pairs:
					batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5])
					test_res = evaluate_tree(config, test_batch[0], test_batch[1], generate_num_ids, embedding, encoder, predict, generate,
											 merge, input_lang, output_lang, test_batch[4], test_batch[5], batch_graph, test_batch[7], beam_size=config.beam_size)
					val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
					
					cur_result = 0
					if val_ac:
						value_ac += 1
						cur_result = 1
					if equ_ac:
						equation_ac += 1
					eval_total += 1

					with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
						f_out.write('Example: ' + str(ex_num) + '\n')
						f_out.write('Source: ' + stack_to_string(sentence_from_indexes(input_lang, test_batch[0])) + '\n')
						f_out.write('Target: ' + stack_to_string(sentence_from_indexes(output_lang, test_batch[2])) + '\n')
						f_out.write('Generated: ' + stack_to_string(sentence_from_indexes(output_lang, test_res)) + '\n')
						if config.nums_disp:
							src_nums = len(test_batch[4])
							tgt_nums = 0
