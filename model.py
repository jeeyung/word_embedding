import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from dataloader import *
import time 

class generator(nn.Module):
    def __init__(self, char_num, gen_embed_dim, hidden_size, num_layer, dropout, bidirectional, multigpu, device):
        super(generator, self).__init__()
        self.multigpu = multigpu
        self.device = device
        self.embedding = nn.Embedding(char_num, gen_embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(gen_embed_dim, hidden_size, num_layers=num_layer,
                    dropout=dropout, batch_first = True, bidirectional=bidirectional)
                
    def sorting(self, x, x_len):
        x_ordered = np.sort(x_len)[::-1]
        sort_idx = np.argsort(x_len)[::-1]
        unsort_idx = np.argsort(sort_idx)[::-1]
        x_ordered = torch.from_numpy(x_ordered.copy()).to(torch.cuda.current_device())
        sort_idx= torch.from_numpy(sort_idx.copy()).to(torch.cuda.current_device())
        unsort_idx = torch.from_numpy(unsort_idx.copy()).to(torch.cuda.current_device())
        x = x.index_select(0, sort_idx)
        return x, unsort_idx, x_ordered

    def forward(self, x, x_len):
        if self.multigpu:
            total_len = x.size(0)*4
            if torch.cuda.current_device() == 0:
                x_len=x_len[:1*int(total_len/4)]
            elif torch.cuda.current_device() == 1:
                x_len=x_len[1*int(total_len/4):2*int(total_len/4)]
            elif torch.cuda.current_device() == 2:
                x_len=x_len[2*int(total_len/4):3*int(total_len/4)]
            else:
                x_len=x_len[3*int(total_len/4):total_len]
        x, unsort_idx, x_ordered = self.sorting(x, x_len)
        embedded = self.embedding(x)
        embedded = pack_padded_sequence(embedded, x_ordered, batch_first = True)
        _, (h,_) = self.lstm(embedded)
        ordered_output = h[-1].index_select(0, unsort_idx)
        return ordered_output

class skipgram(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(skipgram, self).__init__()
        self.center_embedding = nn.Embedding(vocab_size, embed_size)
        self.context_embedding = nn.Embedding(vocab_size, embed_size)

    def pos_loss(self, center, context):
        score_target = torch.bmm(center.unsqueeze(1), context.unsqueeze(2))
        loss = F.logsigmoid(score_target).sum()
        return loss

    def neg_loss(self, center, ns):
        score_neg = torch.bmm(center.unsqueeze(1), ns.transpose(1, 2))
        loss = F.logsigmoid(-score_neg).sum()
        return loss

    def forward(self, center, context, ns):
        center = self.center_embedding(center)
        context = self.context_embedding(context)
        ns = self.context_embedding(ns) #[32,5,128]
        return -self.pos_loss(center, context) + -self.neg_loss(center, ns)

class word_embed_ng(nn.Module):
    def __init__(self, char_num, gen_embed_dim, hidden_size, num_layer, dropout, last_hidden, k, bidirectional, multigpu, device):
        super(word_embed_ng, self).__init__()
        self.center_generator = generator(char_num, gen_embed_dim, hidden_size, num_layer, dropout, bidirectional, multigpu, device)
        self.context_generator = generator(char_num, gen_embed_dim, hidden_size, num_layer, dropout, bidirectional, multigpu, device)
        self.mlp = nn.Linear(hidden_size, last_hidden)
        self.k = k

    def cal_loss(self, x, y, neg):
        score_target = torch.bmm(x.unsqueeze(1),y.unsqueeze(2))
        score_neg = torch.bmm(x.unsqueeze(1), neg.transpose(0,1).transpose(1,2))
        loss = -F.logsigmoid(score_target).sum() + -F.logsigmoid(-score_neg).sum()
        return loss

    def forward(self, x, x_len, y, y_len, neg):
        prediction = self.mlp(self.center_generator(x, x_len))
        target = self.mlp(self.context_generator(y, y_len))
        neg_output =[]
        for i in range(self.k):
            neg_output.append(self.mlp(self.context_generator(neg[i][0], neg[i][1])))
        neg_output_tensor = torch.stack(neg_output)
        loss = self.cal_loss(prediction, target, neg_output_tensor)
        return loss

if __name__=='__main__':

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = word_embed_ng(26, 10, 10, 1, 0.3, 10, 5, False, True, device)
    # model = model.to(device)
    
    # text_loader = TextDataLoader('./data', batch_size = 2, window_size = 5, k=5)
    text_loader = TextDataLoader('./data', 'toy/merge.txt', 16, 5, 5, True, 0, 5, 1e-04)
    
    for i, (center, context, neg) in enumerate(text_loader):
        center, center_len = center
        context, context_len = context
        center = center.to(device)
        context = context.to(device)
        n =[]
        for k in range(5):
            padded_neg, neg_len = neg[k]
            n.append((padded_neg.to(device), neg_len))
        if torch.cuda.device_count() > 1:
            print("using", torch.cuda.device_count(), "GPUs")
            model = nn.DataParallel(model)
        model = model.to(device)
        output = model(center, center_len, context, context_len, neg)
        print(output)
        break
        