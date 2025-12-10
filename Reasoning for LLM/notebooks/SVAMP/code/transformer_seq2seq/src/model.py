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
			)
		elif self.config.opt == 'asgd':
			self.optimizer = optim.ASGD(
				[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
				{"params": self.non_emb_params, "lr": self.config.lr}]
			)
		else:
			self.optimizer = optim.SGD(
				[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
				{"params": self.non_emb_params, "lr": self.config.lr}]
			)

	def generate_square_subsequent_mask(self, sz):
		'''
			Args:
				sz (integer): max_len of sequence in target without EOS i.e. (T-1)
			Returns:
				mask (tensor) : square mask | size : [T-1 x T-1]
		'''

		mask = torch.triu(torch.ones(sz, sz), 1)
		mask = mask.masked_fill(mask==1, float('-inf'))
		return mask

	def make_len_mask(self, inp):
		'''
			Args:
				inp (tensor): input indices | size : [S x BS]
			Returns:
				mask (tensor) : pad mask | size : [BS x S]
		'''

		mask = (inp == -1).transpose(0, 1)
		return mask
		# return (inp == self.EOS_token).transpose(0, 1)

	def forward(self, ques, src, trg):
		'''
			Args:
				ques (list): raw source input | size : [BS]
				src (tensor): source indices | size : [S x BS]
				trg (tensor): target indices | size : [T x BS]
			Returns:
				output (tensor) : Network output | size : [T-1 x BS x voc2.nwords]
		'''

		if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
			self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

		# trg_mask when T-1 = 4: [When decoding for position i, only indexes with 0 in the ith row are attended over]
		# tensor([[0., -inf, -inf, -inf],
		# 		[0., 0., -inf, -inf],
		# 		[0., 0., 0., -inf],
		# 		[0., 0., 0., 0.],

		if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
			src, src_tokens = self.embedding1(ques)
			src = src.transpose(0,1)
			# src: Tensor [S x BS x d_model]
			src_pad_mask = self.make_len_mask(src_tokens.transpose(0,1))
			src = self.pos_embedding1(src)
		else:
			src_pad_mask = self.make_len_mask(src)
			src = self.embedding1(src)
			src = self.pos_embedding1(src)

		trg_pad_mask = self.make_len_mask(trg)
		trg = self.embedding2(trg)
		trg = self.pos_embedding2(trg)

		output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask, memory_mask=self.memory_mask,
								  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)
		
		output = self.fc_out(output)

		return output

	def trainer(self, ques, input_seq1, input_seq2, config, device=None ,logger=None):
		'''
			Args:
				ques (list): raw source input | size : [BS]
				input_seq1 (tensor): source indices | size : [S x BS]
				input_seq2 (tensor): target indices | size : [T x BS]
			Returns:
				fin_loss (float) : Train Loss
		'''

		self.optimizer.zero_grad() # zero out gradients from previous backprop computations

		output = self.forward(ques, input_seq1, input_seq2[:-1,:])
		# output: (T-1) x BS x voc2.nwords [T-1 because it predicts after start symbol]

		output_dim = output.shape[-1]
		
		self.loss = self.criterion(output.view(-1, output_dim), input_seq2[1:,:].view(-1))

		self.loss.backward()
		if self.config.max_grad_norm > 0:
			torch.nn.utils.clip_grad_norm_(self.params, self.config.max_grad_norm)
		self.optimizer.step()

		fin_loss = self.loss.item()

		return fin_loss

