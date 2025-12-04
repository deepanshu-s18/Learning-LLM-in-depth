import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
import pdb

class BertEncoder(nn.Module):
	def __init__(self, bert_model = 'bert-base-uncased',device = 'cuda:0 ', freeze_bert = False):
		super(BertEncoder, self).__init__()
		self.bert_layer = BertModel.from_pretrained(bert_model)
		self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
		self.device = device
		
		if freeze_bert:
			for p in self.bert_layer.parameters():
				p.requires_grad = False
		
	def bertify_input(self, sentences):
		'''
		Preprocess the input sentences using bert tokenizer and converts them to a torch tensor containing token ids
		
		Args:
			sentences (list): source sentences
		Returns:
			token_ids (tensor): tokenized sentences | size: [BS x S]
			attn_masks (tensor): masks padded indices | size: [BS x S]
			input_lengths (list): lengths of sentences | size: [BS]
		'''

		# Tokenize the input sentences for feeding into BERT
		all_tokens  = [['[CLS]'] + self.bert_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]
		
		# Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		# Convert tokens to token ids
		token_ids = torch.tensor([self.bert_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		# Obtain attention masks
		pad_token = self.bert_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a BERT encoder to obtain contextualized representations of each token
