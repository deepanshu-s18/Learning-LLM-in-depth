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
		packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
		outputs, hidden = self.gru(packed, hidden)
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
		outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
		# S x B x H
		return outputs, hidden


class Attn(nn.Module):
	def __init__(self, hidden_size):
		super(Attn, self).__init__()
		self.hidden_size = hidden_size
		self.attn = nn.Linear(hidden_size * 2, hidden_size)
		self.score = nn.Linear(hidden_size, 1, bias=False)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, hidden, encoder_outputs, seq_mask=None):
		max_len = encoder_outputs.size(0)
		repeat_dims = [1] * hidden.dim()
		repeat_dims[0] = max_len
		hidden = hidden.repeat(*repeat_dims)  # S x B x H
		# For each position of encoder outputs
		this_batch_size = encoder_outputs.size(1)
		energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
		attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
		attn_energies = attn_energies.squeeze(1)
		attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
		if seq_mask is not None:
			attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
		attn_energies = self.softmax(attn_energies)
		# Normalize energies to weights in range 0 to 1, resize to B x 1 x S
		return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
	def __init__(
			self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
		super(AttnDecoderRNN, self).__init__()

		# Keep for reference
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.input_size = input_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout

		# Define layers
		self.em_dropout = nn.Dropout(dropout)
		self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
		self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
		self.concat = nn.Linear(hidden_size * 2, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		# Choose attention model
		self.attn = Attn(hidden_size)

	def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
		# Get the embedding of the current input word (last output word)
		batch_size = input_seq.size(0)
		embedded = self.embedding(input_seq)
		embedded = self.em_dropout(embedded)
		embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N

		# Calculate attention from current RNN state and all encoder outputs;
		# apply to encoder outputs to get weighted average
		attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

		# Get current hidden state from input word and last hidden state
		rnn_output, hidden = self.gru(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

		# Attentional vector using the RNN hidden state and context vector
		# concatenated together (Luong eq. 5)
		output = self.out(torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

		# Return final output, hidden state
		return output, hidden


class TreeNode:  # the class save the tree node
	def __init__(self, embedding, left_flag=False):
		self.embedding = embedding
		self.left_flag = left_flag


class Score(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(Score, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.attn = nn.Linear(hidden_size + input_size, hidden_size)
		self.score = nn.Linear(hidden_size, 1, bias=False)

	def forward(self, hidden, num_embeddings, num_mask=None):
		max_len = num_embeddings.size(1)
		repeat_dims = [1] * hidden.dim()
		repeat_dims[1] = max_len
		hidden = hidden.repeat(*repeat_dims)  # B x O x H
		# For each position of encoder outputs
		this_batch_size = num_embeddings.size(0)
		energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
		score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
		score = score.squeeze(1)
		score = score.view(this_batch_size, -1)  # B x O
		if num_mask is not None:
			score = score.masked_fill_(num_mask, -1e12)
		return score


class TreeAttn(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(TreeAttn, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.attn = nn.Linear(hidden_size + input_size, hidden_size)
		self.score = nn.Linear(hidden_size, 1)

	def forward(self, hidden, encoder_outputs, seq_mask=None):
		max_len = encoder_outputs.size(0)

		repeat_dims = [1] * hidden.dim()
		repeat_dims[0] = max_len
		hidden = hidden.repeat(*repeat_dims)  # S x B x H
		this_batch_size = encoder_outputs.size(1)

		energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

		score_feature = torch.tanh(self.attn(energy_in))
		attn_energies = self.score(score_feature)  # (S x B) x 1
		attn_energies = attn_energies.squeeze(1)
		attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
		if seq_mask is not None:
			attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
		attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

		return attn_energies.unsqueeze(1)


class EncoderSeq(nn.Module):
	# def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
	def __init__(self, cell_type, embedding_size, hidden_size, n_layers=2, dropout=0.5):
		super(EncoderSeq, self).__init__()

		# self.input_size = input_size
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.dropout = dropout

		# self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
		# self.em_dropout = nn.Dropout(dropout)

		if cell_type == 'lstm':
			self.rnn = nn.LSTM(self.embedding_size, self.hidden_size,
							   num_layers=self.n_layers,
							   dropout=(0 if self.n_layers == 1 else self.dropout),
							   bidirectional=True)
		elif cell_type == 'gru':
			self.rnn = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
		else:
			self.rnn = nn.RNN(self.embedding_size, self.hidden_size,
							  num_layers=self.n_layers,
							  nonlinearity='tanh',							# ['relu', 'tanh']
							  dropout=(0 if self.n_layers == 1 else self.dropout),
							  bidirectional=True)

		self.gcn = Graph_Module(hidden_size, hidden_size, hidden_size)

	def forward(self, embedded, input_lengths, orig_idx, batch_graph, hidden=None):
		# Note: we run this all at once (over multiple batches of multiple sequences)
		# embedded = self.embedding(input_seqs)  # S x B x E
		# embedded = self.em_dropout(embedded)
		packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
		pade_hidden = hidden
		# pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
		pade_outputs, pade_hidden = self.rnn(packed, pade_hidden)
		pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

		if orig_idx is not None:
			pade_outputs = pade_outputs.index_select(1, orig_idx)

		problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
		pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
		# pdb.set_trace()
		_, pade_outputs = self.gcn(pade_outputs, batch_graph)
		pade_outputs = pade_outputs.transpose(0, 1)
		return pade_outputs, problem_output


class Prediction(nn.Module):
	# a seq2tree decoder with Problem aware dynamic encoding

	def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
		super(Prediction, self).__init__()

		# Keep for reference
		self.hidden_size = hidden_size
		self.input_size = input_size
		self.op_nums = op_nums

		# Define layers
		self.dropout = nn.Dropout(dropout)

		self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

		# for Computational symbols and Generated numbers
		self.concat_l = nn.Linear(hidden_size, hidden_size)
		self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
