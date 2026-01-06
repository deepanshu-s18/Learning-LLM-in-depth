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
			for num in num_stack:
				if decoder_output[i, num_start + num] > max_score:
					target[i] = num + num_start
					max_score = decoder_output[i, num_start + num]
	return target

def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos):
	# mask the decoder input number and return the mask tensor and the encoder position Hidden vector
	up_num_start = decoder_input >= nums_start
	down_num_end = decoder_input < (nums_start + copy_nums)
	num_mask = up_num_start == down_num_end
	num_mask_encoder = num_mask < 1
	num_mask_encoder = num_mask_encoder.unsqueeze(1)  # ByteTensor size: B x 1
	repeat_dims = [1] * num_mask_encoder.dim()
	repeat_dims[1] = embedding_size
	num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

	all_embedding = encoder_outputs.transpose(0, 1).contiguous()
	all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
	indices = decoder_input - nums_start
	indices = indices * num_mask.long()  # 0 or the num pos in sentence
	indices = indices.tolist()
	for k in range(len(indices)):
		indices[k] = num_pos[k][indices[k]]
	indices = torch.LongTensor(indices)
	if USE_CUDA:
		indices = indices.cuda()
	batch_size = decoder_input.size(0)
	sen_len = encoder_outputs.size(0)
	batch_num = torch.LongTensor(range(batch_size))
	batch_num = batch_num * sen_len
	if USE_CUDA:
		batch_num = batch_num.cuda()
	indices = batch_num + indices
	num_encoder = all_embedding.index_select(0, indices)
	return num_mask, num_encoder, num_mask_encoder

def out_equation(test, output_lang, num_list, num_stack=None):
	test = test[:-1]
	max_index = len(output_lang.index2word) - 1
	test_str = ""
	for i in test:
		if i < max_index:
			c = output_lang.index2word[i]
			if c == "^":
				test_str += "**"
			elif c == "[":
				test_str += "("
			elif c == "]":
				test_str += ")"
			elif c[0] == "N":
				if int(c[1:]) >= len(num_list):
					return None
				x = num_list[int(c[1:])]
				if x[-1] == "%":
					test_str += "(" + x[:-1] + "/100" + ")"
				else:
					test_str += x
			else:
				test_str += c
		else:
			if len(num_stack) == 0:
				print(test_str, num_list)
				return ""
			n_pos = num_stack.pop()
			test_str += num_list[n_pos[0]]
	return test_str

def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
	# print(test_res, test_tar)

	if len(num_stack) == 0 and test_res == test_tar:
		return True, True, test_res, test_tar
	test = out_expression_list(test_res, output_lang, num_list)
	tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
	# print(test, tar)
	if test is None:
		return False, False, test, tar
	if test == tar:
		return True, True, test, tar
	try:
		if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
			return True, False, test, tar
		else:
			return False, False, test, tar
	except:
		return False, False, test, tar

def compute_postfix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
	# print(test_res, test_tar)

	if len(num_stack) == 0 and test_res == test_tar:
		return True, True, test_res, test_tar
	test = out_expression_list(test_res, output_lang, num_list)
	tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
