import os
import sys
import math
import logging
import pdb
import random
from time import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_pretrained_bert.optimization import BertAdam
# from tensorboardX import SummaryWriter
from gensim import models
from src.components.encoder import Encoder
from src.components.decoder import DecoderRNN
from src.components.attention import LuongAttnDecoderRNN
from src.components.bert_encoder import BertEncoder
from src.utils.sentence_processing import *
from src.utils.logger import print_log, store_results
from src.utils.helper import save_checkpoint, bleu_scorer
from src.utils.evaluate import cal_score, stack_to_string, get_infix_eq
from src.confidence_estimation import *
from collections import OrderedDict



class Seq2SeqModel(nn.Module):
	def __init__(self, config, voc1, voc2, device, logger, num_iters, EOS_tag='</s>', SOS_tag='<s>'):
		super(Seq2SeqModel, self).__init__()

		self.config = config
		self.device = device
		self.voc1 = voc1
		self.voc2 = voc2
		self.EOS_tag = EOS_tag
		self.SOS_tag = SOS_tag
		self.EOS_token = voc2.get_id(EOS_tag)
		self.SOS_token = voc2.get_id(SOS_tag)
		self.logger = logger
		self.num_iters = num_iters

		# To Do: Embeddings from pretrained models like BERT
		# if self.config.use_word2vec:
		# 	config.emb1_size = 300
		# 	self.embedding1  = nn.Embedding.from_pretrained(torch.FloatTensor(self._form_embeddings(self.config.word2vec_bin)), freeze=not self.config.train_word2vec)
		# else:
		# 	self.embedding1  = nn.Embedding(self.voc1.nwords, self.config.emb1_size)
		# 	nn.init.uniform_(self.embedding1.weight, -1 * self.config.init_range, self.config.init_range)
		# # self.embedding = nn.Embedding(self.voc.nwords, self.config.emb_size)

		self.embedding2  = nn.Embedding(self.voc2.nwords, self.config.emb2_size)
		nn.init.uniform_(self.embedding2.weight, -1 * self.config.init_range, self.config.init_range)

		self.bert = BertEncoder(config.bert_name, self.device)
		# To Do: Vary initialization methods

		self.logger.debug('Building Encoders...')
		self.encoder = Encoder(
			self.config.hidden_size,
			self.config.bert_size,
			self.config.cell_type,
			self.config.depth,
			self.config.dropout,
			self.config.bidirectional
		)
		

		self.logger.debug('Encoders Built...')

		if self.config.use_attn:
			self.decoder    = LuongAttnDecoderRNN(self.config.attn_type,
												  self.embedding2,
												  self.config.cell_type,
												  self.config.hidden_size,
												  self.voc2.nwords,
												  self.config.depth,
												  self.config.dropout).to(device)
		else:
			self.decoder    = DecoderRNN(self.embedding2,
										 self.config.cell_type,
										 self.config.hidden_size,
										 self.voc2.nwords,
										 self.config.depth,
										 self.config.dropout).to(device)



		self.logger.debug('Decoder RNN Built...')


		self.logger.debug('Initalizing Optimizer and Criterion...')
		self._initialize_optimizer()

		# nn.CrossEntropyLoss() does both F.log_softmax() and nn.NLLLoss() 
		self.criterion = nn.NLLLoss() 

		self.logger.info('All Model Components Initialized...')

	def _form_embeddings(self, file_path):
		weights_all = models.KeyedVectors.load_word2vec_format(file_path, limit=200000, binary=True)
		weight_req  = torch.randn(self.voc1.nwords, self.config.emb1_size)
		for key, value in self.voc1.id2w.items():
			if value in weights_all:
				weight_req[key] = torch.FloatTensor(weights_all[value])

		return weight_req	

	def _initialize_optimizer(self):
		self.params = list(self.encoder.parameters()) + \
			list(self.decoder.parameters())

		param_optimizer = list(self.bert.get_model().named_parameters())
		no_decay = ['bias', 'gamma', 'beta']

		optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
		]

		self.bertoptimizer = BertAdam(optimizer_grouped_parameters,
						 lr=self.config.bert_lr,
						 warmup=self.config.warmup,
						 t_total=self.num_iters*self.config.epochs) 

		if self.config.opt == 'adam':
			self.optimizer = optim.Adam(self.params, lr=self.config.lr)
		elif self.config.opt == 'adadelta':
			self.optimizer = optim.Adadelta(self.params, lr=self.config.lr)
		elif self.config.opt == 'asgd':
			self.optimizer = optim.ASGD(self.params, lr=self.config.lr)
		else:
			self.optimizer = optim.SGD(self.params, lr=self.config.lr)

	# def encode(self, input_seqs, input_len, encoder):
	# 	pdb.set_trace()
	# 	output, hidden = encoder(input_seqs, input_len)
	# 	return output

	# def clf(self, hidden):
	# 	return self.soft_clf(hidden)

	def forward(self, input_seq1, input_seq2, input_len1, input_len2):
		'''
			Args:
				input_seq1 (tensor): values are word indexes | size : [max_len x batch_size]
				input_len1 (tensor): Length of each sequence in input_len1 | size : [batch_size]
				input_seq2 (tensor): values are word indexes | size : [max_len x batch_size]
				input_len2 (tensor): Length of each sequence in input_len2 | size : [batch_size]
			Returns:
				out (tensor) : Probabilities of each output label for each point | size : [batch_size x num_labels]
		'''


	def trainer(self, ques, input_seq2, input_len2, config, device=None ,logger=None):

		self.optimizer.zero_grad()
		self.bertoptimizer.zero_grad()

		input_seq1, input_len1 = self.bert(ques)
		input_seq1 = input_seq1.transpose(0,1)

		encoder_outputs, encoder_hidden = self.encoder(input_seq1, input_len1, self.device)
		
		self.loss =0

		decoder_input = torch.tensor([self.SOS_token for i in range(input_seq1.size(1))], device = self.device)

		if config.cell_type == 'lstm':
			decoder_hidden = (encoder_hidden[0][:self.decoder.nlayers], encoder_hidden[1][:self.decoder.nlayers])
		else:
			decoder_hidden = encoder_hidden[:self.decoder.nlayers]

		use_teacher_forcing = True if random.random() < self.config.teacher_forcing_ratio else False
		target_len = max(input_len2)

		if use_teacher_forcing:
			for step in range(target_len):
				if self.config.use_attn:
					decoder_output, decoder_hidden, decoder_attention, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
				else:
					decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
				self.loss += self.criterion(decoder_output, input_seq2[step])
				decoder_input = input_seq2[step]
		else:
			for step in range(target_len):
				if self.config.use_attn:
					decoder_output, decoder_hidden, decoder_attention, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
				else:
					decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
				
				topv, topi = decoder_output.topk(1)
				self.loss += self.criterion(decoder_output, input_seq2[step])
				decoder_input = topi.squeeze().detach() 


		self.loss.backward()
		if self.config.max_grad_norm > 0:
			torch.nn.utils.clip_grad_norm_(self.params, self.config.max_grad_norm)
		self.optimizer.step()
		self.bertoptimizer.step()

		return self.loss.item()/target_len

	# def evaluator(self, input_seq1, input_seq2, input_len1, input_len2, config, device=None ,logger=None):
		



	# 	return acc/batch_size, loss/batch_size
