import gensim.models.keyedvectors as word2vec
import torch
import torch.nn as nn
import sys
sys.path.append("evaluation")
print(sys.path)
import torch
from dataloader import TestDataset
from model import *
from dataloader import *
from evaluate import evaluate

google_path = "/media/hk/D/GoogleNews-vectors-negative300.bin"
fasttext_path = "/media/hk/D/wiki-news-300d-1M-subword.vec"

def pre_embedding(path):
    if "bin" in path:
        model = word2vec.KeyedVectors.load_word2vec_format(path, binary = True)
    else:
        model = word2vec.KeyedVectors.load_word2vec_format(path)
    word2index = {token: token_index for token_index, token in enumerate(model.index2word)}
    weights = torch.FloatTensor(model.wv.vectors)
    embeddings = nn.Embedding.from_pretrained(weights)
    return embeddings, word2index

google_skipgram, word2index = pre_embedding(google_path)
fasttext_model, word2index2 = pre_embedding(fasttext_path)

evaluation_google_skipgram = evaluate(google_skipgram, True, word2idx = word2index)
evaluation_fasttext = evaluate(fasttext_model, True, word2idx = word2index2)