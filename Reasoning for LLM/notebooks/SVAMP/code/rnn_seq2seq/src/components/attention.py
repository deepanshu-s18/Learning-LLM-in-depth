import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

# Luong attention layer
class Attn(nn.Module):
	def __init__(self, method, hidden_size):
		super(Attn, self).__init__()
		self.method = method
		if self.method not in ['dot', 'general', 'concat']:
			raise ValueError(self.method, "is not an appropriate attention method.")
		self.hidden_size = hidden_size
		if self.method == 'general':
			self.attn = nn.Linear(self.hidden_size, hidden_size)
		elif self.method == 'concat':
			self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
			self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

	def dot_score(self, hidden, encoder_outputs):
		return torch.sum(hidden * encoder_outputs, dim=2)

	def general_score(self, hidden, encoder_outputs):
		energy = self.attn(encoder_outputs)
