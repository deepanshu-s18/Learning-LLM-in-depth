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
			self.config.embedding_size = 300
			self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self._form_embeddings(self.config.word2vec_bin)), freeze = self.config.freeze_emb)
		else:
			self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
		self.em_dropout = nn.Dropout(dropout)

	def _form_embeddings(self, file_path):
		weights_all = models.KeyedVectors.load_word2vec_format(file_path, limit=200000, binary=True)
		weight_req  = torch.randn(self.input_size, self.config.embedding_size)
		for temp_ind in range(len(self.input_lang.index2word)):
			value = self.input_lang.index2word[temp_ind]
			if value in weights_all:
				weight_req[temp_ind] = torch.FloatTensor(weights_all[value])
		# for key, value in self.voc1.id2w.items():
		# 	if value in weights_all:
		# 		weight_req[key] = torch.FloatTensor(weights_all[value])

		return weight_req

	def forward(self, input_seqs):
		embedded = self.embedding(input_seqs)  # S x B x E
		embedded = self.em_dropout(embedded)
		return embedded

class EncoderRNN(nn.Module):
	def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
		super(EncoderRNN, self).__init__()

		self.input_size = input_size
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.dropout = dropout

		self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
		self.em_dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

	def forward(self, input_seqs, input_lengths, hidden=None):
		# Note: we run this all at once (over multiple batches of multiple sequences)
		embedded = self.embedding(input_seqs)  # S x B x E
		embedded = self.em_dropout(embedded)
