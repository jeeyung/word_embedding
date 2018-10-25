import sys, os
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re
import random 
import operator
from functools import reduce
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class news_dataset(Dataset):
    def __init__(self, window_size, data_dir, k=5):
        self.data_dir = os.path.join(data_dir, 'toy')
        self.char2idx = self.map_char_idx()
        self.k = k
        self.word_pairs = self.make_pairs(window_size)

    def map_char_idx(self):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        char2idx = {}
        for i in range(len(alphabet)):
            char2idx[list(alphabet)[i]] = i+1
        return char2idx

    def make_pairs(self, window_size):
        text_path = os.path.join(self.data_dir, 'merge.txt')
        self.tokenized_text = self.tokenize(text_path)
        self.tokenized_text_flatten = reduce(operator.concat, self.tokenized_text)
        self.vocab = list(set(self.tokenized_text_flatten))
        # print(len(self.tokenized_text_flatten), len(self.vocab))
        word_pairs = []
        for sentence in self.tokenized_text:
            for i, word in enumerate(sentence):
                # start_time = time.time()
                sentence = np.asarray(sentence)
                for w in range(-window_size, window_size + 1):
                    context_word_idx = i + w
                    if context_word_idx < 0 or context_word_idx >= len(sentence)\
                        or context_word_idx == i:
                        continue
                    neg_samples = self.negative_sampling()
                    word_pairs.append((word, sentence[context_word_idx], neg_samples))
                    # word_pairs.append((word, sentence[context_word_idx]))
                # total_time = time.time() - start_time
                # print(total_time)
        return word_pairs
                
    def tokenize(self, text_path):
        text = open(text_path, encoding="utf-8").read().lower().strip()
        text = re.sub('[^A-Za-z]+', " ", text)
        text = text.split(".")
        tokens = [sen.split() for sen in text]
        return tokens
    
    def negative_sampling(self):
        # tokenized_text_flatten = reduce(operator.concat, tokenized_text)
        # print(len(tokenized_text_flatten))
        # neg_sample = np.random.choice(self.vocab, self.k)
        neg_sample = random.sample(self.vocab, self.k)
        return neg_sample
        
    def _unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def make_char(self, pairs):
        center, context, neg_samples = pairs
        center_idx = [self.char2idx[char] for char in list(center)]
        context_idx = [self.char2idx[char] for char in list(context)]
        negs_idx = []
        for i in range(self.k):
            negs_idx.append([self.char2idx[char] for char in list(neg_samples[i])])
        return center_idx, context_idx, negs_idx
    
    def __getitem__(self, idx):
        center_idx, context_idx, negs_idx = self.make_char(self.word_pairs[idx])
        center_idx, context_idx = torch.tensor(center_idx, device= device), torch.tensor(context_idx, device = device)
        negs_tensor_idx = []
        for i in range(self.k):
            negs_tensor_idx.append(torch.tensor(negs_idx[i], device=device))
        return center_idx, context_idx, negs_tensor_idx

    def __len__(self):
        return len(self.word_pairs)

    
if __name__ == '__main__':

    text_dataset = news_dataset(2, './data')
    
    index = 1
    center_idx, context_idx = text_dataset[index]
    print(center_idx, context_idx)