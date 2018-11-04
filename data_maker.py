import os
import re
import random
import operator
import numpy as np
import pickle as pkl
from functools import reduce
from configuration import get_config

args = get_config()

def make_dataset(dataset='toy/merge.txt', window_size, ns_size, is_character):
    text_path = os.path.join(args.data_dir, self.dataset)
    tokenized_text = tokenize(text_path)
    tokenized_text_flatten = reduce(operator.concat, tokenized_text)
    vocabs = list(set(tokenized_text_flatten))
    word_pairs = []
    for sentence in tokenized_text:
        for i, word in enumerate(sentence):
            sentence = np.asarray(sentence)
            for w in range(-window_size, window_size + 1):
                context_word_idx = i + w
                if context_word_idx < 0 or context_word_idx >= len(sentence) or context_word_idx == i:
                    continue
                neg_samples = negative_sampling(vocabs, ns_size)
                word_pairs.append((word, sentence[context_word_idx], neg_samples))
    return word_pairs, vocabs

def make_char(pairs):
    center, context, neg_samples = pairs
    center_idx = [self.char2idx[char] for char in list(center)]
    context_idx = [self.char2idx[char] for char in list(context)]
    negs_idx = []
    for i in range(self.ns_size):
        negs_idx.append([self.char2idx[char] for char in list(neg_samples[i])])
    return center_idx, context_idx, negs_idx


def map_char_idx():
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    char2idx = {}
    idx2char = {}
    for i in range(len(alphabet)):
        char2idx[list(alphabet)[i]] = i + 1
        idx2char[i + 1] = list(alphabet)[i]
    return char2idx, idx2char

def map_word_idx(vocab):
    word2idx = {}
    idx2word = {}
    for n, word in enumerate(vocab):
        word2idx[word] = n
        idx2word[n] = word
    return word2idx, idx2word

def tokenize(text_path):
    text = open(text_path, encoding="utf-8").read().lower().strip()
    text = re.sub('[^A-Za-z]+', " ", text)
    text = text.split(".")
    tokens = [sen.split() for sen in text]
    return tokens


def negative_sampling(vocab, ns_size):
    neg_sample = random.sample(vocab, ns_size)
    return neg_sample


def _unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


if __name__ == '__main__':
    pairs, vocabs = make_news_dataset(5, 5)
    print(pairs[0])