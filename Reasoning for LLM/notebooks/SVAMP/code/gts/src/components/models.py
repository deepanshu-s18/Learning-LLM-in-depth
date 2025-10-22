import torch
import torch.nn as nn
from gensim import models
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

