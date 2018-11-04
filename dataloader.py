import sys, os
import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader
from dataset import TextDataset


def collate_text(list_inputs):
    print(list_inputs)
    batch = len(list_inputs)
    center_list = [len(list_inputs[i][0]) for i in range(batch)]
    max_len_center = max(center_list)
    padded_center = torch.zeros(batch, max_len_center, dtype=torch.long)
    context_list = [len(list_inputs[i][1]) for i in range(batch)]
    max_len_context = max(context_list)
    padded_context = torch.zeros(batch, max_len_context, dtype=torch.long)
    for i in range(batch):
        padded_center[i,:center_list[i]] = list_inputs[i][0]
        padded_context[i, :context_list[i]] = list_inputs[i][1]
    neg = []
    neg_size = len(list_inputs[0][2])
    for k in range(neg_size):
        neg_len = [len(list_inputs[i][2][k]) for i in range(batch)]
        max_len_neg = max(neg_len)
        padded_neg = torch.zeros(batch, max_len_neg, dtype=torch.long)
        for i in range(batch):
            padded_neg[i,:neg_len[i]] = list_inputs[i][2][k]
        neg.append((padded_neg, neg_len))
    return (padded_center, center_list), (padded_context, context_list), neg

# def collate_text_ng(list_inputs):
#     batch = len(list_inputs)
#     padded_center, center_list = create_list(list_inputs[i,0], batch)
#     padded_context, context_list = create_list(list_inputs[:,1], batch)
#     neg = []
#     for i in range(7):
#         neg.append((create_list(list_inputs[:,2][i], batch)))
#     return padded_center, center_list, padded_context, context_list, neg
#
# def create_list(raw_input, batch):
#     input_len = [len(raw_input[i][0]) for i in range(batch)]
#     max_len = max(input_len)
#     padded_input = torch.zeros(batch, max_len, dtype = torch.long, device = device)
#     for i in range(batch):
#         padded_input[i, :input_len[i]] = raw_input[i]
#     return padded_input, input_len

class TextDataLoader(DataLoader):
    def __init__(self, data_dir, dataset, batch_size, window_size, ns_size, is_character):
        self.dataset = TextDataset(data_dir, dataset, window_size, ns_size, is_character)
        if is_character:
            super(TextDataLoader, self).__init__(self.dataset, batch_size, collate_fn=collate_text)
        else:
            super(TextDataLoader, self).__init__(self.dataset, batch_size)


if __name__ == '__main__':
    is_character = False
    text_loader = TextDataLoader('./data', 'toy/merge.txt', 10, 5, 5, is_character)
    if is_character:
        for i, (padded_center, center_list, padded_context, context_list, neg) in enumerate(text_loader):
            print(padded_center)
            print(center_list)
            print(padded_context)
            print(context_list)
            print(neg)
            if i > 10:
                break
    else:
        for i, (center_word, context_word, ns_words) in enumerate(text_loader):
            print(center_word)
            print(context_word)
            print(ns_words)
            if i > 10:
                break