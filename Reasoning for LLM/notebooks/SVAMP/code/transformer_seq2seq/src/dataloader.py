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
		Expecting csv files with columns ['Question', 'Equation', 'Numbers', 'Answer']

		Args:
						data_path: Root folder Containing all the data
						dataset: Specific Folder ==> data_path/dataset/	(Should contain train.csv and dev.csv)
						max_length: Self Explanatory
						is_debug: Load a subset of data for faster testing
						is_train: 

	'''

