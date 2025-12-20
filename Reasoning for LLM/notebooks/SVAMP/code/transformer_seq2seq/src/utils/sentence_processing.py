import logging
import pdb
import torch
from glob import glob
from torch.autograd import Variable
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def sent_to_idx(voc, sent, max_length, flag = 0):
	if flag == 0:
		idx_vec = []
	else:
		idx_vec = [voc.get_id('<s>')]
	for w in sent.split(' '):
		try:
			idx = voc.get_id(w)
			idx_vec.append(idx)
		except:
			idx_vec.append(voc.get_id('unk'))
	# idx_vec.append(voc.get_id('</s>'))
	if flag == 1 and len(idx_vec) < max_length-1:
		idx_vec.append(voc.get_id('</s>'))
	return idx_vec

def sents_to_idx(voc, sents, max_length, flag = 0):
	all_indexes = []
	for sent in sents:
		all_indexes.append(sent_to_idx(voc, sent, max_length, flag))
	return all_indexes

def sent_to_tensor(voc, sentence, device, max_length):
	indexes = sent_to_idx(voc, sentence, max_length)
	return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def batch_to_tensor(voc, sents, device, max_length):
	batch_sent = []
	# batch_label = []
	for sent in sents:
		sent_id = sent_to_tensor(voc, sent, device, max_length)
		batch_sent.append(sent_id)

	return batch_sent

