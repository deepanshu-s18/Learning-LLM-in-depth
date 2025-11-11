import os
import sys
import math
import logging
import pdb
import random
from time import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_pretrained_bert.optimization import BertAdam
# from tensorboardX import SummaryWriter
from gensim import models
from src.components.encoder import Encoder
from src.components.decoder import DecoderRNN
from src.components.attention import LuongAttnDecoderRNN
from src.components.bert_encoder import BertEncoder
from src.utils.sentence_processing import *
from src.utils.logger import print_log, store_results
from src.utils.helper import save_checkpoint, bleu_scorer
from src.utils.evaluate import cal_score, stack_to_string, get_infix_eq
from src.confidence_estimation import *
from collections import OrderedDict



class Seq2SeqModel(nn.Module):
	def __init__(self, config, voc1, voc2, device, logger, num_iters, EOS_tag='</s>', SOS_tag='<s>'):
		super(Seq2SeqModel, self).__init__()

		self.config = config
		self.device = device
		self.voc1 = voc1
		self.voc2 = voc2
		self.EOS_tag = EOS_tag
		self.SOS_tag = SOS_tag
		self.EOS_token = voc2.get_id(EOS_tag)
		self.SOS_token = voc2.get_id(SOS_tag)
