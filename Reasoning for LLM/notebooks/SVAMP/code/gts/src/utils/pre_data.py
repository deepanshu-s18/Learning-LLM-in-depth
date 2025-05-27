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
