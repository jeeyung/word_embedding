import sys, os
import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader
from dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_text(list_inputs):
    batch = len(list_inputs)
    center_list = [len(list_inputs[i][0]) for i in range(batch)]
    max_len_center = max(center_list)
    padded_center = torch.zeros(batch, max_len_center, dtype = torch.long, device=device)
    context_list = [len(list_inputs[i][1]) for i in range(batch)]
    max_len_context = max(context_list)
    padded_context = torch.zeros(batch, max_len_context, dtype = torch.long, device=device)
    for i in range(batch):
        padded_center[i,:center_list[i]] = list_inputs[i][0]
        padded_context[i, :context_list[i]] = list_inputs[i][1]
    neg = []
    neg_size = len(list_inputs[0][2])
    for k in range(neg_size):
        neg_len = [len(list_inputs[i][2][k]) for i in range(batch)]
        max_len_neg = max(neg_len)
        padded_neg = torch.zeros(batch, max_len_neg, dtype= torch.long, device=device)
        for i in range(batch):
            padded_neg[i,:neg_len[i]] = list_inputs[i][2][k]
        neg.append((padded_neg, neg_len))
    return padded_center, center_list, padded_context, context_list, neg

def collate_text_ng(list_inputs):
    batch = len(list_inputs)
    padded_center, center_list = create_list(list_inputs[i,0], batch)
    padded_context, context_list = create_list(list_inputs[:,1], batch)
    neg = []
    for i in range(7):
        neg.append((create_list(list_inputs[:,2][i], batch)))
    return padded_center, center_list, padded_context, context_list, neg

def create_list(raw_input, batch):
    input_len = [len(raw_input[i][0]) for i in range(batch)]
    max_len = max(input_len)
    padded_input = torch.zeros(batch, max_len, dtype = torch.long, device = device)
    for i in range(batch):
        padded_input[i, :input_len[i]] = raw_input[i]
    return padded_input, input_len

class TextDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, window_size, k):
        self.batch_size = batch_size
        self.dataset = news_dataset(window_size, data_dir, k)
        super(TextDataLoader, self).__init__(self.dataset, self.batch_size, collate_fn=collate_text)
        


if __name__ == '__main__':

    text_loader = TextDataLoader('./data', 10, 5)
    
    for i, (pc, cl, pco, col) in enumerate(text_loader):
        # print(data)
        print(pc)
        print(cl)
        print(pco)
        print(col)
    #     # padded = pad_packed_sequence(target)
        break