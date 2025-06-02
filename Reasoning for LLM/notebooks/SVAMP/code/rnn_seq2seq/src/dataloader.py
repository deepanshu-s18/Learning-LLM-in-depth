import os
import logging
import pdb
import re
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import unicodedata
from collections import OrderedDict

class TextDataset(Dataset):
	'''
		Expecting csv files with columns ['sent1', 'sent2']

		Args:
						data_path: Root folder Containing all the data
						dataset: Specific Folder==> data_path/dataset/	(Should contain train.csv and dev.csv)
						max_length: Self Explanatory
						is_debug: Load a subset of data for faster testing
						is_train: 

	'''

	def __init__(self, data_path='./data/', dataset='mawps', datatype='train', max_length=30, is_debug=False, is_train=False, grade_info=False, type_info=False, challenge_info=False):
		if datatype=='train':
			file_path = os.path.join(data_path, dataset, 'train.csv')
		elif datatype=='dev':
			file_path = os.path.join(data_path, dataset, 'dev.csv')
		else:
			file_path = os.path.join(data_path, dataset, 'dev.csv')

		if grade_info:
			self.grade_info = True
		else:
			self.grade_info = False

		if type_info:
			self.type_info = True
		else:
			self.type_info = False

		if challenge_info:
			self.challenge_info = True
		else:
			self.challenge_info = False

		file_df= pd.read_csv(file_path)
