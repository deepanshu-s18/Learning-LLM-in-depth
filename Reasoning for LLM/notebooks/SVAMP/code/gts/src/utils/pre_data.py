import random
import json
import copy
import re
import os
import pandas as pd
import nltk
import pdb

PAD_token = 0

class Lang:
	"""
	class to save the vocab and two dict: the word->index and index->word
	"""
	def __init__(self):
		self.word2index = {}
		self.word2count = {}
		self.index2word = []
		self.n_words = 0  # Count word tokens
		self.num_start = 0

	def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
		for word in sentence:
			if re.search("N\d+|NUM|\d+", word):
				continue
			if word not in self.index2word:
				self.word2index[word] = self.n_words
				self.word2count[word] = 1
				self.index2word.append(word)
				self.n_words += 1
			else:
				self.word2count[word] += 1

	def trim(self, logger, min_count):  # trim words below a certain count threshold
		keep_words = []

		for k, v in self.word2count.items():
			if v >= min_count:
				keep_words.append(k)

		logger.debug('keep_words {} / {} = {}'.format(len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)))

		# Reinitialize dictionaries
		self.word2index = {}
		# self.word2count = {}
		self.index2word = []
		self.n_words = 0  # Count default tokens

		for word in keep_words:
			self.word2index[word] = self.n_words
			self.index2word.append(word)
			self.n_words += 1

	def build_input_lang(self, logger, trim_min_count):  # build the input lang vocab and dict
		if trim_min_count > 0:
			self.trim(logger, trim_min_count)
			self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
		else:
			self.index2word = ["PAD", "NUM"] + self.index2word
		self.word2index = {}
		self.n_words = len(self.index2word)
		for i, j in enumerate(self.index2word):
			self.word2index[j] = i

	def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
		self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
						  ["SOS", "UNK"]
		self.n_words = len(self.index2word)
		for i, j in enumerate(self.index2word):
			self.word2index[j] = i

	def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
		self.num_start = len(self.index2word)

		self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
		self.n_words = len(self.index2word)

		for i, j in enumerate(self.index2word):
			self.word2index[j] = i

def load_raw_data(data_path, dataset, is_train = True):  # load the data to list(dict())
	train_ls = None
	if is_train:
		train_path = os.path.join(data_path, dataset, 'train.csv')
		train_df = pd.read_csv(train_path)
		train_ls = train_df.to_dict('records')

	dev_path = os.path.join(data_path, dataset, 'dev.csv')
	dev_df = pd.read_csv(dev_path)
	dev_ls = dev_df.to_dict('records')

	return train_ls, dev_ls

# remove the superfluous brackets
def remove_brackets(x):
	y = x
	if x[0] == "(" and x[-1] == ")":
		x = x[1:-1]
		flag = True
		count = 0
		for s in x:
			if s == ")":
				count -= 1
				if count < 0:
