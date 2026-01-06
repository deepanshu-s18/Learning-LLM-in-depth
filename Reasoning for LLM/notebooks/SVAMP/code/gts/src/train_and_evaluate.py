import math
import torch
import torch.optim
import torch.nn.functional as f
import time
import pdb

from src.components.models import *
from src.components.masked_cross_entropy import *
from src.utils.pre_data import *
from src.utils.expressions_transfer import *
from src.utils.helper import *

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()

class Beam:  # the class save the beam node
	def __init__(self, score, input_var, hidden, all_output):
		self.score = score
		self.input_var = input_var
		self.hidden = hidden
		self.all_output = all_output

def time_since(s):  # compute time
	m = math.floor(s / 60)
	s -= m * 60
	h = math.floor(m / 60)
	m -= h * 60
	return '%dh %dm %ds' % (h, m, s)

def generate_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums, generate_nums,
					   english):
	rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
	if english:
		if decoder_input[0] == word2index["SOS"]:
			for i in range(batch_size):
				res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
					  [word2index["("]] + generate_nums
				for j in res:
					rule_mask[i, j] = 0
			return rule_mask
		for i in range(batch_size):
			res = []
			if decoder_input[i] >= nums_start:
				res += [word2index[")"], word2index["+"], word2index["-"],
						word2index["/"], word2index["*"], word2index["EOS"]
						]
			elif decoder_input[i] in generate_nums:
				res += [word2index[")"], word2index["+"], word2index["-"],
						word2index["/"], word2index["*"], word2index["EOS"]
						]
			elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
				res += [PAD_token]
			elif decoder_input[i] == word2index["("]:
				res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
				  [word2index["("]] + generate_nums
			elif decoder_input[i] == word2index[")"]:
				res += [word2index[")"], word2index["+"], word2index["-"],
						word2index["/"], word2index["*"], word2index["EOS"]
						]
			elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
				res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + [word2index["("]] + generate_nums
			for j in res:
				rule_mask[i, j] = 0
	else:
		if decoder_input[0] == word2index["SOS"]:
			for i in range(batch_size):
				res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
					  [word2index["["], word2index["("]] + generate_nums
				for j in res:
					rule_mask[i, j] = 0
			return rule_mask
		for i in range(batch_size):
			res = []
			if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
				res += [word2index["]"], word2index[")"], word2index["+"],
						word2index["-"], word2index["/"], word2index["^"],
						word2index["*"], word2index["EOS"]
						]
			elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
				res += [PAD_token]
			elif decoder_input[i] == word2index["["] or decoder_input[i] == word2index["("]:
				res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
				  [word2index["("]] + generate_nums
			elif decoder_input[i] == word2index[")"]:
				res += [word2index["]"], word2index[")"], word2index["+"],
						word2index["-"], word2index["/"], word2index["^"],
						word2index["*"], word2index["EOS"]
						]
			elif decoder_input[i] == word2index["]"]:
				res += [word2index["+"], word2index["*"], word2index["-"], word2index["/"], word2index["EOS"]]
			elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"],
									  word2index["*"], word2index["^"]]:
				res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
				  [word2index["["], word2index["("]] + generate_nums
			for j in res:
				rule_mask[i, j] = 0
	return rule_mask


def generate_pre_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
									generate_nums, english):
	rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
	if english:
		if decoder_input[0] == word2index["SOS"]:
			for i in range(batch_size):
				res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
					  [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
				for j in res:
					rule_mask[i, j] = 0
			return rule_mask
		for i in range(batch_size):
			res = []
			if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
				res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
					   [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
						]
			elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
				res += [PAD_token]
			elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
				res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
					   [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
			for j in res:
				rule_mask[i, j] = 0
	else:
		if decoder_input[0] == word2index["SOS"]:
			for i in range(batch_size):
				res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
					  [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
				for j in res:
					rule_mask[i, j] = 0
			return rule_mask
		for i in range(batch_size):
			res = []
			if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
				res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
					   [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"],
						word2index["^"]
						]
			elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
				res += [PAD_token]
			elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
									  word2index["^"]]:
				res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
					   [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
			for j in res:
				rule_mask[i, j] = 0
	return rule_mask


def generate_post_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
									 generate_nums, english):
	rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
	if english:
		if decoder_input[0] == word2index["SOS"]:
			for i in range(batch_size):
				res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
				for j in res:
					rule_mask[i, j] = 0
			return rule_mask
		for i in range(batch_size):
			res = []
			if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
				res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
					   [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
			elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
				res += [PAD_token]
			elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
				res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums +\
					   [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
						]
			for j in res:
				rule_mask[i, j] = 0
	else:
		if decoder_input[0] == word2index["SOS"]:
			for i in range(batch_size):
				res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
				for j in res:
					rule_mask[i, j] = 0
			return rule_mask
		for i in range(batch_size):
			res = []
			if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
				res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
					   [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]
						]
			elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
				res += [PAD_token]
			elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
									  word2index["^"]]:
				res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
					   [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"],
						word2index["EOS"]
						]
			for j in res:
				rule_mask[i, j] = 0
	return rule_mask

def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
	# when the decoder input is copied num but the num has two pos, chose the max
	target_input = copy.deepcopy(target)
	for i in range(len(target)):
		if target[i] == unk:
			num_stack = nums_stack_batch[i].pop()
			max_score = -float("1e12")
			for num in num_stack:
				if decoder_output[i, num_start + num] > max_score:
					target[i] = num + num_start
					max_score = decoder_output[i, num_start + num]
		if target_input[i] >= num_start:
			target_input[i] = 0
	return torch.LongTensor(target), torch.LongTensor(target_input)

def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
	# when the decoder input is copied num but the num has two pos, chose the max
	if USE_CUDA:
		decoder_output = decoder_output.cpu()
	for i in range(target.size(0)):
		if target[i] == unk:
			num_stack = nums_stack_batch[i].pop()
			max_score = -float("1e12")
