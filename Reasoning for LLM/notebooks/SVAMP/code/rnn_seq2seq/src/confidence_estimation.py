import numpy as np
import torch
from src.utils.bleu import compute_bleu

def posterior_based_conf(test_ques, model):

    decoded_words, decoded_log_probs = model.greedy_decode(test_ques, return_probs = True)
    posteriors = [np.exp(sum(log_probs)) for log_probs in decoded_log_probs]
    return decoded_words, posteriors

def similarity_based_conf(test_ques, train_ques,model, sim_criteria = 'bert_score'):
    '''
    Takes a batch of test question and evaluates their closest similarities between questions in training set.
    Inputs:
        test_ques: A list of strings containing a batch of test questions. Length: Batch Size
        train_ques: A list containing **ALL** the questions present in training data. Length: |Training Data|
        model: bert_seq2exp model
        sim_criteria: Criteria used to evaluate similarity between test questions and training questions

