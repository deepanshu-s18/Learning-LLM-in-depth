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

