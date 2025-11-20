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
from transformers import AdamW
# from tensorboardX import SummaryWriter
from gensim import models
from src.components.encoder import Encoder
from src.components.decoder import DecoderRNN
from src.components.attention import LuongAttnDecoderRNN
from src.components.contextual_embeddings import BertEncoder, RobertaEncoder
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

		self.embedding2 = nn.Embedding(self.voc2.nwords, self.config.emb2_size)
		nn.init.uniform_(self.embedding2.weight, -1 * self.config.init_range, self.config.init_range)

		if self.config.embedding == 'bert':
			self.embedding1 = BertEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'roberta':
			self.embedding1 = RobertaEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'word2vec':
			self.config.emb1_size = 300
			self.embedding1 = nn.Embedding.from_pretrained(torch.FloatTensor(self._form_embeddings(self.config.word2vec_bin)), freeze = self.config.freeze_emb)
		else:
			self.embedding1  = nn.Embedding(self.voc1.nwords, self.config.emb1_size)
			nn.init.uniform_(self.embedding1.weight, -1 * self.config.init_range, self.config.init_range)

		self.logger.debug('Building Encoders...')
		self.encoder = Encoder(
			self.config.hidden_size,
			self.config.emb1_size,
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
		self.params =   list(self.embedding1.parameters()) + \
						list(self.encoder.parameters()) + \
						list(self.decoder.parameters())

		if self.config.separate_opt:
			self.emb_optimizer = AdamW(self.embedding1.parameters(), lr = self.config.emb_lr, correct_bias = True)
			self.optimizer = optim.Adam(
				[{"params": self.encoder.parameters()},
				{"params": self.decoder.parameters()}],
				lr = self.config.lr,
			)
		else:
			if self.config.opt == 'adam':
				self.optimizer = optim.Adam(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.encoder.parameters()},
					{"params": self.decoder.parameters()}],
					lr = self.config.lr
				)
			elif self.config.opt == 'adadelta':
				self.optimizer = optim.Adadelta(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.encoder.parameters()},
					{"params": self.decoder.parameters()}],
					lr = self.config.lr
				)
			elif self.config.opt == 'asgd':
				self.optimizer = optim.ASGD(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.encoder.parameters()},
					{"params": self.decoder.parameters()}],
					lr = self.config.lr
				)
			else:
				self.optimizer = optim.SGD(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.encoder.parameters()},
					{"params": self.decoder.parameters()}],
					lr = self.config.lr
				)

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

	def trainer(self, ques, input_seq1, input_seq2, input_len1, input_len2, config, device=None ,logger=None):
		'''
			Args:
				ques (list): input examples as is (i.e. not indexed) | size : [batch_size]
			Returns:
				
		'''
		self.optimizer.zero_grad()
		if self.config.separate_opt:
			self.emb_optimizer.zero_grad()

		if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
			input_seq1, input_len1 = self.embedding1(ques)
			input_seq1 = input_seq1.transpose(0,1)
			# input_seq1: Tensor [max_len x BS x emb1_size]
			# input_len1: List [BS]
			sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
			# sorted_seqs: Tensor [max_len x BS x emb1_size]
			# input_len1: List [BS]
			# orig_idx: Tensor [BS]
		else:
			sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
			sorted_seqs = self.embedding1(sorted_seqs)

		encoder_outputs, encoder_hidden = self.encoder(sorted_seqs, sorted_len, orig_idx, self.device)
		
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
		if self.config.separate_opt:
			self.emb_optimizer.step()

		return self.loss.item()/target_len

	def greedy_decode(self, ques, input_seq1=None, input_seq2=None, input_len1=None, input_len2=None, validation=False, return_probs = False):
		with torch.no_grad():
			if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
				input_seq1, input_len1 = self.embedding1(ques)
				input_seq1 = input_seq1.transpose(0,1)
				sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
			else:
				sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
				sorted_seqs = self.embedding1(sorted_seqs)

			encoder_outputs, encoder_hidden = self.encoder(sorted_seqs, sorted_len, orig_idx, self.device)

			loss = 0.0
			decoder_input = torch.tensor([self.SOS_token for i in range(input_seq1.size(1))], device=self.device)

			if self.config.cell_type == 'lstm':
				decoder_hidden = (encoder_hidden[0][:self.decoder.nlayers], encoder_hidden[1][:self.decoder.nlayers])
			else:
				decoder_hidden = encoder_hidden[:self.decoder.nlayers]

			decoded_words = [[] for i in range(input_seq1.size(1))]
			decoded_probs = [[] for i in range(input_seq1.size(1))]
			decoder_attentions = []

			if validation:
				target_len = max(input_len2)
			else:
				target_len = self.config.max_length

			for step in range(target_len):
				if self.config.use_attn:
					decoder_output, decoder_hidden, decoder_attention, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
					decoder_attentions.append(decoder_attention)
				else:
					decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

				if validation:
					loss += self.criterion(decoder_output, input_seq2[step])
				topv, topi = decoder_output.topk(1)
				for i in range(input_seq1.size(1)):
					if topi[i].item() == self.EOS_token:
						continue
					decoded_words[i].append(self.voc2.get_word(topi[i].item()))
					decoded_probs[i].append(topv[i].item())
				decoder_input = topi.squeeze().detach()

			if validation:
				if self.config.use_attn:
					return loss/target_len, decoded_words, decoder_attentions[:step + 1]
				else:
					return loss/target_len, decoded_words, None
			else:
				if return_probs:
					return decoded_words, decoded_probs

				return decoded_words

	def obtain_hidden(self, config, ques, input_seq1=None, input_seq2=None, input_len1=None, input_len2=None):
		with torch.no_grad():
			if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
				input_seq1, input_len1 = self.embedding1(ques)
				input_seq1 = input_seq1.transpose(0,1)
				sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
			else:
				sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
				sorted_seqs = self.embedding1(sorted_seqs)

			encoder_outputs, encoder_hidden = self.encoder(sorted_seqs, sorted_len, orig_idx, self.device)

			loss =0.0
			decoder_input = torch.tensor([self.SOS_token for i in range(input_seq1.size(1))], device=self.device)

			if self.config.cell_type == 'lstm':
				decoder_hidden = (encoder_hidden[0][:self.decoder.nlayers], encoder_hidden[1][:self.decoder.nlayers])
			else:
				decoder_hidden = encoder_hidden[:self.decoder.nlayers]

			decoded_words = [[] for i in range(input_seq1.size(1))]
			decoder_attentions = []

			hiddens = []

			target_len = max(input_len2)

			for step in range(target_len):
				if self.config.use_attn:
					decoder_output, decoder_hidden, decoder_attention, hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
					decoder_attentions.append(decoder_attention)
				else:
					decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

				topv, topi = decoder_output.topk(1)
				for i in range(input_seq1.size(1)):
					if topi[i].item() == self.EOS_token:
						continue
					decoded_words[i].append(self.voc2.get_word(topi[i].item()))
					hiddens.append([self.voc2.get_word(topi[i].item()), hidden[i]])
				decoder_input = topi.squeeze().detach()

			return hiddens, decoded_words

def build_model(config, voc1, voc2, device, logger, num_iters):
	'''
		Add Docstring
	'''
	model = Seq2SeqModel(config, voc1, voc2, device, logger, num_iters)
	model = model.to(device)

	return model

def train_model(model, train_dataloader, val_dataloader, voc1, voc2, device, config, logger, epoch_offset= 0, min_val_loss=float('inf'), max_val_bleu=0.0, max_val_acc = 0.0, min_train_loss=float('inf'), max_train_acc = 0.0, best_epoch = 0, writer= None):
	'''
		Add Docstring
	'''

	if config.histogram and config.save_writer and writer:
		for name, param in model.named_parameters():
			writer.add_histogram(name, param, epoch_offset)
	
	estop_count=0
	
	for epoch in range(1, config.epochs + 1):
		od = OrderedDict()
		od['Epoch'] = epoch + epoch_offset
		print_log(logger, od)

		batch_num = 1
		train_loss_epoch = 0.0
		train_acc_epoch = 0.0
		train_acc_epoch_cnt = 0.0
		train_acc_epoch_tot = 0.0
		val_loss_epoch = 0.0

		start_time= time()
		total_batches = len(train_dataloader)

