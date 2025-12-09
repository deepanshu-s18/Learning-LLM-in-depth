import os
import sys
import math
import logging
import pdb
import random
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW
# from pytorch_pretrained_bert.optimization import BertAdam
# from tensorboardX import SummaryWriter
from gensim import models
from src.components.contextual_embeddings import BertEncoder, RobertaEncoder
from src.utils.sentence_processing import *
from src.utils.logger import print_log, store_results
from src.utils.helper import save_checkpoint, bleu_scorer
from src.utils.evaluate import cal_score, stack_to_string, get_infix_eq

from collections import OrderedDict

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.scale = nn.Parameter(torch.ones(1)) # nn.Parameter causes the tensor to appear in the model.parameters()

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # max_len x 1
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # torch.arange(0, d_model, 2) gives 2i
		pe[:, 0::2] = torch.sin(position * div_term) # all alternate columns 0 onwards
		pe[:, 1::2] = torch.cos(position * div_term) # all alternate columns 1 onwards
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		'''
			Args:
				x (tensor): embeddings | size : [max_len x batch_size x d_model]
			Returns:
				z (tensor) : embeddings with positional encoding | size : [max_len x batch_size x d_model]
		'''
		
		x = x + self.scale * self.pe[:x.size(0), :]
		z = self.dropout(x)
		return z

class TransformerModel(nn.Module):
	def __init__(self, config, voc1, voc2, device, logger, EOS_tag = '</s>', SOS_tag = '<s>'):
		super(TransformerModel, self).__init__()
		self.config = config
		self.device = device
		self.voc1 = voc1
		self.voc2 = voc2
		self.EOS_tag = EOS_tag
		self.SOS_tag = SOS_tag
		self.EOS_token = voc2.get_id(EOS_tag)
		self.SOS_token = voc2.get_id(SOS_tag)
		self.logger = logger

		self.logger.debug('Initialising Embeddings.....')

		if self.config.embedding == 'bert':
			config.d_model = 768
			self.embedding1 = BertEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'roberta':
			config.d_model = 768
			self.embedding1 = RobertaEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'word2vec':
			config.d_model = 300
			self.embedding1  = nn.Embedding.from_pretrained(torch.FloatTensor(self._form_embeddings(self.config.word2vec_bin)), 
								freeze = self.config.freeze_emb)
		else:
			self.embedding1  = nn.Embedding(self.voc1.nwords, self.config.d_model)
			nn.init.uniform_(self.embedding1.weight, -1 * self.config.init_range, self.config.init_range)

		self.pos_embedding1 = PositionalEncoding(self.config.d_model, self.config.dropout)

		self.embedding2  = nn.Embedding(self.voc2.nwords, self.config.d_model)
		nn.init.uniform_(self.embedding2.weight, -1 * self.config.init_range, self.config.init_range)
		
		self.pos_embedding2 = PositionalEncoding(self.config.d_model, self.config.dropout)

		self.logger.debug('Embeddings initialised.....')
		self.logger.debug('Building Transformer Model.....')

		self.transformer = nn.Transformer(d_model=self.config.d_model, nhead=self.config.heads, 
											num_encoder_layers=self.config.encoder_layers, num_decoder_layers=self.config.decoder_layers, 
											dim_feedforward=self.config.d_ff, dropout=self.config.dropout)
		
		self.fc_out = nn.Linear(self.config.d_model, self.voc2.nwords)

		self.logger.debug('Transformer Model Built.....')

		self.src_mask = None
		self.trg_mask = None
		self.memory_mask = None

		self.logger.debug('Initalizing Optimizer and Criterion...')

		self._initialize_optimizer()

		self.criterion = nn.CrossEntropyLoss() # nn.CrossEntropyLoss() does both F.log_softmax() and nn.NLLLoss() 

		self.logger.info('All Model Components Initialized...')

	def _form_embeddings(self, file_path):
		'''
			Args:
				file_path (string): path of file with word2vec weights
			Returns:
				weight_req (tensor) : embedding matrix | size : [voc1.nwords x d_model]
		'''

		weights_all = models.KeyedVectors.load_word2vec_format(file_path, limit=200000, binary=True)
		weight_req  = torch.randn(self.voc1.nwords, self.config.d_model)
		for key, value in self.voc1.id2w.items():
			if value in weights_all:
				weight_req[key] = torch.FloatTensor(weights_all[value])

		return weight_req

	def _initialize_optimizer(self):
		self.params = list(self.embedding1.parameters()) + list(self.transformer.parameters()) + list(self.fc_out.parameters()) + \
						list(self.embedding2.parameters()) + list(self.pos_embedding1.parameters()) + list(self.pos_embedding2.parameters())
		self.non_emb_params = list(self.transformer.parameters()) + list(self.fc_out.parameters()) + list(self.embedding2.parameters()) + \
								list(self.pos_embedding1.parameters()) + list(self.pos_embedding2.parameters())

		if self.config.opt == 'adam':
			self.optimizer = optim.Adam(
				[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
				{"params": self.non_emb_params, "lr": self.config.lr}]
			)
		elif self.config.opt == 'adamw':
			self.optimizer = optim.AdamW(
				[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
				{"params": self.non_emb_params, "lr": self.config.lr}]
			)
		elif self.config.opt == 'adadelta':
			self.optimizer = optim.Adadelta(
				[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
				{"params": self.non_emb_params, "lr": self.config.lr}]
