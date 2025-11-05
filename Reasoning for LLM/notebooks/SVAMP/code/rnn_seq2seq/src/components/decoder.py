import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class DecoderRNN(nn.Module):
	'''
	To DO
	Encoder helps in building the sentence encoding module for a batched version
	of data that is sent in [T x B] having corresponding input lengths in [1 x B]

	Args:
			hidden_size: Hidden size of the RNN cell
			embedding: Embeddings matrix [vocab_size, embedding_dim]
