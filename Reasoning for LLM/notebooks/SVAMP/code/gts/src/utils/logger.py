import logging
import pdb
import pandas as pd
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import json

'''Logging Modules'''

def get_logger(name, log_file_path='./logs/temp.log', logging_level=logging.INFO, log_format='%(asctime)s | %(levelname)s | %(filename)s: %(lineno)s : %(funcName)s() ::\t %(message)s'):
	logger = logging.getLogger(name)
	logger.setLevel(logging_level)
	formatter = logging.Formatter(log_format)

	file_handler = logging.FileHandler(log_file_path, mode='w')
	file_handler.setLevel(logging_level)
	file_handler.setFormatter(formatter)

	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(logging_level)
	stream_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)

	return logger

def print_log(logger, dict):
	string = ''
	for key, value in dict.items():
		string += '\n {}: {}\t'.format(key.replace('_', ' '), value)
	logger.info(string)

def store_results(config, max_train_acc, max_val_acc, eq_acc, min_train_loss, best_epoch):
	try:
		with open(config.result_path) as f:
			res_data =json.load(f)
	except:
		res_data = {}
	try:
		min_train_loss = min_train_loss.item()
	except:
		pass
	# try:
	# 	min_val_loss = min_val_loss.item()
	# except:
	# 	pass
	try:
		data= {'run name' : str(config.run_name)
		, 'max val acc': str(max_val_acc)
		, 'equation acc': str(eq_acc)
		, 'max train acc': str(max_train_acc)
		, 'min train loss': str(min_train_loss)
		, 'best epoch': str(best_epoch)
		, 'epochs' : config.epochs
		, 'dataset' : config.dataset
		, 'embedding': config.embedding
		, 'embedding_size' : config.embedding_size
		, 'embedding_lr': config.emb_lr
		, 'freeze_emb': config.freeze_emb
		, 'cell_type' : config.cell_type
		, 'hidden_size' : config.hidden_size
		, 'depth' : config.depth
