import sys, os
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset, ConcatDataset
import re
import random
import operator
from functools import reduce
from nltk.corpus import stopwords
import bz2
import _pickle as cPickle
from multiprocessing import Process, Pool
import multiprocessing as mp
from collections import Counter
from functools import wraps
import time
import random
# from sklearn.externals import joblib

def timefn(fn):
    def wrap(*args):
        t1 = time.time()
        result = fn(*args)
        t2 = time.time()
        print("@timefn:{} took {} seconds".format(fn.__name__, t2-t1))
        return result
    return wrap

class TextDataset(Dataset):
    def __init__(self, data_dir, dataset, window_size, ns_size, remove_th, subsam_th, is_character):
        self.dataset_dir = os.path.join(data_dir, dataset)
        data_config_list = [dataset, window_size, ns_size, is_character]
        self.data_file = '_'.join(map(str, data_config_list)) + '.pkl'
        self.file_dir = os.path.join(data_dir, self.data_file)
        self.window_size = window_size
        self.ns_size = ns_size
        self.is_character = is_character
        self.rm_th = remove_th
        self.subsam_th = subsam_th
        # self.stopwords = set(stopwords.words('english'))
        self.stopwords = set()
        if not self.is_data_exist():
            self.open_file()

        with open(self.file_dir, 'rb') as f:
            if is_character:
                # self.word_pairs, self.vocabs, self.char2idx, self.idx2char = joblib.load(f)
                self.word_pairs, self.neg_samples, self.vocabs, self.char2idx, self.idx2char = pkl.load(f)
            else:
                # self.word_pairs, self.vocabs, self.word2idx, self.idx2word = joblib.load(f)
                self.word_pairs, self.neg_samples, self.vocabs, self.word2idx, self.idx2word = pkl.load(f)
    
    def open_file(self):
        if self.dataset_dir.endswith(".bz2"):
            text = bz2.BZ2File(self.dataset_dir).read().decode("utf-8").lower().strip()
        else:
            text = open(self.dataset_dir, encoding="utf-8").read().lower().strip()
        self.make_dataset(text)

    def is_data_exist(self):
        if os.path.isfile(self.file_dir):
            print("Data {} exist".format(self.data_file))
            return True
        else:
            print("Data {} does not exist".format(self.data_file))
            return False

    @timefn
    def preprocess_counter(self, tokenized):
        tokenized = reduce(operator.concat, tokenized)
        cnt = Counter(tokenized)
        #remove small words
        small_words = [key for key, value in cnt.items() if value < self.rm_th]
        for word in small_words:
            cnt.pop(word, None)
        #subsample
        prob = 1 - (self.subsam_th / (np.array(list(cnt.values())) / sum(cnt.values())))**0.5
        rm_idx = (np.random.random((len(prob),)) < prob)
        rm_words = np.array(list(cnt.keys()))[rm_idx]
        for word in rm_words:
            cnt.pop(word, None)
        # vocabs
        self.vocabs = list(cnt.keys())
        # calculate distribution
        power = 3 / 4
        probs = (np.array(list(cnt.values())) / sum(cnt.values()))**power
        self.probs = probs / probs.sum()
        self.stopwords = self.stopwords|set(small_words)|set(rm_words)
        return cnt

    def make_dataset(self, text):
        print("Start to make data")
        tokenized_text = self.tokenize(text)
        print("compelete tokenize")
        self.cnt = self.preprocess_counter(tokenized_text)
        print("Start to make data again")
        tokenized_text = self.tokenize(text)
        print("compelete tokenize again")
        if self.is_character:
            self.char2idx, self.idx2char = self.map_char_idx()
        else:
            word2idx, idx2word = self.map_word_idx(self.vocabs)
        word_pairs = []
        for sentence in tokenized_text:
            for i, word in enumerate(sentence):
                sentence = np.asarray(sentence)
                for w in range(-self.window_size, self.window_size + 1):
                    context_word_idx = i + w
                    if context_word_idx < 0 or context_word_idx >= len(sentence) or context_word_idx == i:
                        continue
                    # neg_samples = self.negative_sampling()
                    if self.is_character:
                        word_pairs.append(self.make_chars((word, sentence[context_word_idx])))
                    else:
                        word_pairs.append((word2idx[word], word2idx[sentence[context_word_idx]]))
        neg_samples = self.negative_sampling(len(word_pairs))
        if self.is_character:
            neg_samples = [self.make_char(neg_sample) for neg_sample in neg_samples]
        else:
            neg_samples = [[word2idx[word] for word in neg_sample] for neg_sample in neg_samples]
        if self.is_character:
            saves = word_pairs, neg_samples, self.vocabs, self.char2idx, self.idx2char
        else:
            saves = word_pairs, neg_samples, self.vocabs, word2idx, idx2word
        with open(self.file_dir, 'wb') as f:
            cPickle.dump(saves, f, protocol=2)
            print("Data saved in {}".format(self.data_file))
    
    @timefn
    def tokenize(self, text):
        text = re.sub("(january|febuary|march|april|may|june|july|august|september|october|november|december)", " ", text)
        text = re.sub('<.*>'," ", text)
        text = re.sub('[^A-Za-z.]+', " ", text)
        text = text.split(".")
        tokens_list=[]
        for sen in text:
            tokens=[]
            for word in sen.split():
                if word not in self.stopwords:
                    tokens.append(word)
            tokens_list.append(tokens)
        return tokens_list

    def map_char_idx(self):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        char2idx = {}
        idx2char = {}
        for i in range(len(alphabet)):
            char2idx[list(alphabet)[i]] = i+1
            idx2char[i+1] = list(alphabet)[i]
        return char2idx, idx2char

    def map_word_idx(self, vocabs):
        word2idx = {}
        idx2word = {}
        for n, word in enumerate(vocabs):
            word2idx[word] = n
            idx2word[n] = word
        return word2idx, idx2word

    # @timefn
    def negative_sampling(self, batch):
        # neg_sample = np.random.choice(self.vocabs, self.ns_size, replace=True, p=probs)
        neg_sample = np.random.choice(self.vocabs, size=self.ns_size * batch, replace=True, p=self.probs)
        neg_sample = neg_sample.reshape(batch, self.ns_size)
        return neg_sample

    def _unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

    def make_chars(self, pairs):
        center, context = pairs
        center_idx = [self.char2idx[char] for char in list(center)]
        context_idx = [self.char2idx[char] for char in list(context)]
        return center_idx, context_idx

    def make_char(self, neg_samples):
        negs_idx = []
        for i in range(self.ns_size):
            negs_idx.append([self.char2idx[char] for char in list(neg_samples[i])])
        return negs_idx

    def __getitem__(self, idx):
        if self.is_character:
            center_idx, context_idx = self.word_pairs[idx]
            negs_idx = self.neg_samples[idx]
            center_idx, context_idx = torch.tensor(center_idx), torch.tensor(context_idx)
            negs_tensor_idx = []
            for i in range(self.ns_size):
                negs_tensor_idx.append(torch.tensor(negs_idx[i]))
            return center_idx, context_idx, negs_tensor_idx
        else:
            center_idx, context_idx = self.word_pairs[idx]
            negs_idx = self.neg_samples[idx]
            center_idx, context_idx, negs_idx = torch.tensor(center_idx), torch.tensor(context_idx), torch.tensor(negs_idx)
            return center_idx, context_idx, negs_idx

    def __len__(self):
        return len(self.word_pairs)

def trial(i):
    dataset = 'wiki_{0:02d}.bz2'.format(i)
    print("file", i,"pid=", os.getpid())
    time.sleep(1)

    #for jeeyung
    text_dataset = TextDataset('./data/extracted_wiki/A', dataset, 5, 7, 5, 1e-04, True)
    #for cluster_server
    # text_dataset = TextDataset('/disk2/wiki_dump/A/', dataset, 5, 7, 5, 1e-04, True)
    #for dm_server
    # text_dataset = TextDataset('/data/jeeyung/wiki_dump/B/', dataset, 5, 7, 5, 1e-04, True)

if __name__ == '__main__':
    t1 = time.time()
    text_dataset = TextDataset('./data/extracted_wiki/A', 'wiki_25.bz2', 5, 7, 5, 1e-04, True)
    # text_dataset = TextDataset('./data', 'toy/merge.txt', 5, 7, 5, 1e-04, True)
    t2 = time.time()
    print(t2-t1)
    # index = 1
    # print(text_dataset.word_pairs[index])
    # p = Pool(10)
    # p.map(trial, range(0,100))
    # p.close()
    # p.join()
    
