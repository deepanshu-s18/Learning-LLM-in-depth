import torch
import torch.nn as nn
from gensim import models
import torch.nn.functional as F
import copy
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import pdb

class Embedding(nn.Module):
	def __init__(self, config, input_lang, input_size, embedding_size, dropout=0.5):
		super(Embedding, self).__init__()

		self.config = config
		self.input_lang = input_lang
		self.input_size = input_size
		self.embedding_size = embedding_size

		if self.config.embedding == 'word2vec':
