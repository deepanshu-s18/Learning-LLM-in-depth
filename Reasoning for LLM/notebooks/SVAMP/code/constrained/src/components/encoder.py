import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from src.utils.sentence_processing import sort_by_len, restore_order

class Encoder(nn.Module):
	'''
	Encoder helps in building the sentence encoding module for a batched version
	of data that is sent in [T x B] having corresponding input lengths in [1 x B]

	Args:
			hidden_size: Hidden size of the RNN cell
			embedding: Embeddings matrix [vocab_size, embedding_dim]
			cell_type: Type of RNN cell to be used : LSTM, GRU
			nlayers: Number of layers of LSTM (default = 1)
			dropout: Dropout Rate (default = 0.1)
			bidirectional: Bidirectional model to be formed (default: False)
	'''

	def __init__(self, hidden_size=512,embedding_size = 768, cell_type='lstm', nlayers=1, dropout=0.1, bidirectional=True):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.nlayers = nlayers
		self.dropout = dropout
		self.cell_type = cell_type
		self.embedding_size = embedding_size
		# self.embedding_size = self.embedding.embedding_dim
		self.bidirectional = bidirectional

		if self.cell_type == 'lstm':
			self.rnn = nn.LSTM(self.embedding_size, self.hidden_size,
							   num_layers=self.nlayers,
							   dropout=(0 if self.nlayers == 1 else dropout),
							   bidirectional=bidirectional)
		elif self.cell_type == 'gru':
			self.rnn = nn.GRU(self.embedding_size, self.hidden_size,
							  num_layers=self.nlayers,
							  dropout=(0 if self.nlayers == 1 else dropout),
							  bidirectional=bidirectional)
