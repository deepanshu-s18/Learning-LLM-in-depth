import random
import json
import copy
import re
import numpy as np
import os
import pandas as pd
import nltk
import pdb

PAD_token = 0


class Lang:
	"""
	class to save the vocab and two dict: the word->index and index->word
	"""
	def __init__(self):
		self.word2index = {}
		self.word2count = {}
		self.index2word = []
		self.n_words = 0  # Count word tokens
		self.num_start = 0

	def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
		for word in sentence:
			if re.search("N\d+|NUM|\d+", word):
				continue
			if word not in self.index2word:
				self.word2index[word] = self.n_words
				self.word2count[word] = 1
				self.index2word.append(word)
				self.n_words += 1
			else:
				self.word2count[word] += 1

	def trim(self, min_count):  # trim words below a certain count threshold
		keep_words = []

		for k, v in self.word2count.items():
			if v >= min_count:
				keep_words.append(k)

		print('keep_words %s / %s = %.4f' % (
			len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
		))

		# Reinitialize dictionaries
		self.word2index = {}
		# self.word2count = {}
		self.index2word = []
		self.n_words = 0  # Count default tokens

		for word in keep_words:
			self.word2index[word] = self.n_words
			self.index2word.append(word)
			self.n_words += 1

	def build_input_lang(self, logger, trim_min_count):  # build the input lang vocab and dict
		if trim_min_count > 0:
			self.trim(trim_min_count)
			self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
		else:
			self.index2word = ["PAD", "NUM"] + self.index2word
		self.word2index = {}
		self.n_words = len(self.index2word)
		for i, j in enumerate(self.index2word):
			self.word2index[j] = i

	def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
		self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
						  ["SOS", "UNK"]
		self.n_words = len(self.index2word)
		for i, j in enumerate(self.index2word):
			self.word2index[j] = i

	def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
		self.num_start = len(self.index2word)

		self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
		self.n_words = len(self.index2word)

		for i, j in enumerate(self.index2word):
			self.word2index[j] = i

def load_raw_data(data_path, dataset, is_train = True):  # load the data to list(dict())
	train_ls = None
	if is_train:
		train_path = os.path.join(data_path, dataset, 'train.csv')
		train_df = pd.read_csv(train_path, converters={'group_nums': eval})
		train_ls = train_df.to_dict('records')

	dev_path = os.path.join(data_path, dataset, 'dev.csv')
	dev_df = pd.read_csv(dev_path, converters={'group_nums': eval})
	dev_ls = dev_df.to_dict('records')

	return train_ls, dev_ls


# remove the superfluous brackets
def remove_brackets(x):
	y = x
	if x[0] == "(" and x[-1] == ")":
		x = x[1:-1]
		flag = True
		count = 0
		for s in x:
			if s == ")":
				count -= 1
				if count < 0:
					flag = False
					break
			elif s == "(":
				count += 1
		if flag:
			return x
	return y


def load_mawps_data(filename):  # load the json data to list(dict()) for MAWPS
	print("Reading lines...")
	f = open(filename, encoding="utf-8")
	data = json.load(f)
	out_data = []
	for d in data:
		if "lEquations" not in d or len(d["lEquations"]) != 1: # Only single equations
			continue
		x = d["lEquations"][0].replace(" ", "")

		if "lQueryVars" in d and len(d["lQueryVars"]) == 1: # When Equations are annotated with variables
			v = d["lQueryVars"][0]
			if v + "=" == x[:len(v)+1]: # If eqn of the form 'Var=...'  
				xt = x[len(v)+1:]
				if len(set(xt) - set("0123456789.+-*/()")) == 0:
					temp = d.copy()
					temp["lEquations"] = xt
					out_data.append(temp)
					continue

			if "=" + v == x[-len(v)-1:]: # If eqn of the form '...=Var'
				xt = x[:-len(v)-1]
				if len(set(xt) - set("0123456789.+-*/()")) == 0:
					temp = d.copy()
					temp["lEquations"] = xt
					out_data.append(temp)
					continue

		if len(set(x) - set("0123456789.+-*/()=xX")) != 0: # If equation has anything not in the set on RHS of -
			continue

		if x[:2] == "x=" or x[:2] == "X=":
			if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
				temp = d.copy()
				temp["lEquations"] = x[2:]
				out_data.append(temp)
				continue
		if x[-2:] == "=x" or x[-2:] == "=X":
			if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
				temp = d.copy()
				temp["lEquations"] = x[:-2]
				out_data.append(temp)
				continue
	return out_data


def load_roth_data(filename):  # load the json data to dict(dict()) for roth data
	print("Reading lines...")
	f = open(filename, encoding="utf-8")
	data = json.load(f)
	out_data = {}
	for d in data:
		if "lEquations" not in d or len(d["lEquations"]) != 1:
			continue
		x = d["lEquations"][0].replace(" ", "")

		if "lQueryVars" in d and len(d["lQueryVars"]) == 1:
			v = d["lQueryVars"][0]
			if v + "=" == x[:len(v)+1]:
				xt = x[len(v)+1:]
				if len(set(xt) - set("0123456789.+-*/()")) == 0:
					temp = d.copy()
					temp["lEquations"] = remove_brackets(xt)
					y = temp["sQuestion"]
					seg = y.strip().split(" ")
					temp_y = ""
					for s in seg:
						if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
							temp_y += s[:-1] + " " + s[-1:] + " "
						else:
							temp_y += s + " "
					temp["sQuestion"] = temp_y[:-1]
					out_data[temp["iIndex"]] = temp
					continue

			if "=" + v == x[-len(v)-1:]:
				xt = x[:-len(v)-1]
				if len(set(xt) - set("0123456789.+-*/()")) == 0:
					temp = d.copy()
					temp["lEquations"] = remove_brackets(xt)
					y = temp["sQuestion"]
					seg = y.strip().split(" ")
					temp_y = ""
					for s in seg:
						if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
							temp_y += s[:-1] + " " + s[-1:] + " "
						else:
							temp_y += s + " "
					temp["sQuestion"] = temp_y[:-1]
					out_data[temp["iIndex"]] = temp
					continue

		if len(set(x) - set("0123456789.+-*/()=xX")) != 0:
			continue

		if x[:2] == "x=" or x[:2] == "X=":
			if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
				temp = d.copy()
				temp["lEquations"] = remove_brackets(x[2:])
				y = temp["sQuestion"]
				seg = y.strip().split(" ")
				temp_y = ""
				for s in seg:
					if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
						temp_y += s[:-1] + " " + s[-1:] + " "
					else:
						temp_y += s + " "
				temp["sQuestion"] = temp_y[:-1]
				out_data[temp["iIndex"]] = temp
				continue
		if x[-2:] == "=x" or x[-2:] == "=X":
			if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
				temp = d.copy()
				temp["lEquations"] = remove_brackets(x[2:])
				y = temp["sQuestion"]
				seg = y.strip().split(" ")
				temp_y = ""
				for s in seg:
					if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
						temp_y += s[:-1] + " " + s[-1:] + " "
					else:
						temp_y += s + " "
				temp["sQuestion"] = temp_y[:-1]
				out_data[temp["iIndex"]] = temp
				continue
	return out_data


def transfer_num(train_ls, dev_ls, chall = False):  # transfer num into "NUM"
	print("Transfer numbers...")
	dev_pairs = []
	generate_nums = []
	generate_nums_dict = {}
	copy_nums = 0

	if train_ls != None:
		train_pairs = []
		for d in train_ls:
			# nums = []
			nums = d['Numbers'].split()
			input_seq = []
			seg = nltk.word_tokenize(d["Question"].strip())
			equation = d["Equation"].split()

			numz = ['0','1','2','3','4','5','6','7','8','9']
			opz = ['+', '-', '*', '/']
			idxs = []
			for s in range(len(seg)):
				if len(seg[s]) >= 7 and seg[s][:6] == "number" and seg[s][6] in numz:
					input_seq.append("NUM")
					idxs.append(s)
				else:
					input_seq.append(seg[s])
			if copy_nums < len(nums):
				copy_nums = len(nums)

			out_seq = []
			for e1 in equation:
				if len(e1) >= 7 and e1[:6] == "number":
					out_seq.append('N'+e1[6:])
				elif e1 not in opz:
					generate_nums.append(e1)
					if e1 not in generate_nums_dict:
						generate_nums_dict[e1] = 1
					else:
						generate_nums_dict[e1] += 1
					out_seq.append(e1)
				else:
					out_seq.append(e1)
