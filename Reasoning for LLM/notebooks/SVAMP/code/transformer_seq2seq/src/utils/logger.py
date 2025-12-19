import logging
import pdb
import pandas as pd
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import json

'''Logging Modules'''

#log_format='%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s - %(funcName)5s() ] | %(message)s'
def get_logger(name, log_file_path='./logs/temp.log', logging_level=logging.INFO, 
				log_format='%(asctime)s | %(levelname)s | %(filename)s: %(lineno)s : %(funcName)s() ::\t %(message)s'):
	logger = logging.getLogger(name)
	logger.setLevel(logging_level)
	formatter = logging.Formatter(log_format)

	file_handler = logging.FileHandler(log_file_path, mode='w') # Sends logging output to a disk file
	file_handler.setLevel(logging_level)
	file_handler.setFormatter(formatter)

	stream_handler = logging.StreamHandler() # Sends logging output to stdout
	stream_handler.setLevel(logging_level)
	stream_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)

	# logger.addFilter(ContextFilter(expt_name))

	return logger


def print_log(logger, dict):
	string = ''
	for key, value in dict.items():
		string += '\n {}: {}\t'.format(key.replace('_', ' '), value)
	# string = string.strip()
	logger.info(string)



def store_results(config, max_val_bleu, max_val_acc, min_val_loss, max_train_acc, min_train_loss, best_epoch):
	try:
		with open(config.result_path) as f:
			res_data =json.load(f)
	except:
		res_data = {}
	try:
		min_train_loss = min_train_loss.item()
	except:
		pass
	try:
		min_val_loss = min_val_loss.item()
	except:
		pass
	try:

		data= {'run_name' : str(config.run_name)
		, 'max val acc': str(max_val_acc)
		, 'max train acc': str(max_train_acc)
		, 'max val bleu' : str(max_val_bleu)
		, 'min val loss' : str(min_val_loss)
		, 'min train loss': str(min_train_loss)
		, 'best epoch': str(best_epoch)
		, 'epochs' : config.epochs
		, 'dataset' : config.dataset
		, 'embedding': config.embedding
		, 'embedding_lr': config.emb_lr
		, 'freeze_emb': config.freeze_emb
		, 'i/p and o/p embedding size' : config.d_model
		, 'encoder_layers' : config.encoder_layers
		, 'decoder_layers' : config.decoder_layers
