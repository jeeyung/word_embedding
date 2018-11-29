import sys, os
import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.data import DataLoader
from dataset import TextDataset, TestDataset, PretrainedDataset
from random import Random
from torch import distributed, nn
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
def collate_text(list_inputs):
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

def collate_word(words):
    words.sort(key=lambda x: len(x), reverse=True)
    length_list = [len(words[i]) for i in range(len(words))]
    padded_words = pad_sequence(words).transpose(0,1)
    return padded_words, length_list

def collate_pretrained(list_inputs):
    batch = len(list_inputs)
    words = [input[0] for input in list_inputs]
    words_len = [len(word) for word in words]
    embeddings = torch.cat([input[1].unsqueeze(0) for input in list_inputs], 0)
    max_len = max(words_len)
    padded_words = torch.zeros(batch, max_len, dtype=torch.long)
    for i in range(batch):
        padded_words[i, :words_len[i]] = words[i]
    return padded_words, words_len, embeddings

class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index
         
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataParitioner(object):
    def __init__(self, data, sizes, seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]
    
    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

class TextDataLoader(DataLoader):
    def __init__(self, data_dir, dataset, batch_size, window_size, ns_size, is_character, num_workers, remove_th, subsample_th, multinode):
        self.dataset = TextDataset(data_dir, dataset, window_size, ns_size, remove_th, subsample_th, is_character)
        self.vocabs = self.dataset.vocabs
        self.word2idx = self.dataset.word2idx
        self.char2idx = self.dataset.char2idx
        if multinode:
            size = distributed.get_world_size()
            batch_size = int(batch_size / float(size))
            partition_sizes = [1.0 / size for _ in range(size)]
            partition = DataParitioner(self.dataset, partition_sizes)
            self.dataset = partition.use(distributed.get_rank())
        if is_character:
            super(TextDataLoader, self).__init__(self.dataset, batch_size, num_workers=num_workers, collate_fn=collate_text, shuffle=True)
        else:
            super(TextDataLoader, self).__init__(self.dataset, batch_size, num_workers=num_workers, shuffle=True)

class TestDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size):
        self.dataset = TestDataset(data_dir)
        super(TestDataLoader, self).__init__(self.dataset, batch_size, collate_fn=collate_word)

class PretrainedDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size):
        self.dataset = PretrainedDataset(data_dir)
        super(PretrainedDataLoader, self).__init__(self.dataset, batch_size, collate_fn=collate_pretrained)


if __name__ == '__main__':
    # is_character = False
    # text_loader = TextDataLoader('./data', 'toy/merge.txt', 10, 5, 5, is_character)
    # if is_character:
    #     for i, (padded_center, center_list, padded_context, context_list, neg) in enumerate(text_loader):
    #         print(padded_center)
    #         print(center_list)
    #         print(padded_context)
    #         print(context_list)
    #         print(neg)
    #         if i > 10:
    #             break
    # else:
    #     for i, (center_word, context_word, ns_words) in enumerate(text_loader):
    #         print(center_word)
    #         print(context_word)
    #         print(ns_words)
    #         if i > 10:
    #             break
    test_loader = TestDataLoader('./data', 12, 1)
    for data in test_loader:
        print(data)
