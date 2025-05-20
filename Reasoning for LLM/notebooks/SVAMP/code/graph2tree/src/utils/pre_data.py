import random
import json
import copy
import re
import numpy as np
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

	def trim(self, min_count):  # trim words below a certain count threshold
		keep_words = []

		for k, v in self.word2count.items():
			if v >= min_count:
				keep_words.append(k)

		print('keep_words %s / %s = %.4f' % (
			len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
		))

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
			self.trim(trim_min_count)
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
