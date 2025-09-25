import logging
import pdb
import torch
from glob import glob
from torch.autograd import Variable
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def sent_to_idx(voc, sent, max_length):
	idx_vec = []
	for w in sent.split(' '):
		try:
			idx = voc.get_id(w)
			idx_vec.append(idx)
		except:
			idx_vec.append(voc.get_id('unk'))
	# idx_vec.append(voc.get_id('</s>'))
	if len(idx_vec) < max_length-1:
		idx_vec.append(voc.get_id('</s>'))
	return idx_vec

