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
