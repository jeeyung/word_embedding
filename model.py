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
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            else: 
                nn.init.xavier_normal_(param)

    def sorting(self, x, x_len):
        x_ordered = np.sort(x_len)[::-1]
        sort_idx = np.argsort(x_len)[::-1]
        unsort_idx = np.argsort(sort_idx)[::-1]
        x_ordered = torch.from_numpy(x_ordered.copy()).to(self.device)
        sort_idx= torch.from_numpy(sort_idx.copy()).to(self.device)
        # sort_idx= torch.from_numpy(sort_idx.copy())
        unsort_idx = torch.from_numpy(unsort_idx.copy()).to(self.device)
        x = x.index_select(0, sort_idx)
        return x, unsort_idx, x_ordered

    def forward(self, x, x_len):
        x, unsort_idx, x_ordered = self.sorting(x, x_len)
        embedded = self.embedding(x)
        embedded = pack_padded_sequence(embedded, x_ordered, batch_first = True)
        h_total, (h,_) = self.lstm(embedded)
        ordered_output = h[-1].index_select(0, unsort_idx)
        output, _ = pad_packed_sequence(h_total, batch_first=True)
        h_total_output = output.index_select(0, unsort_idx)
        return ordered_output, h_total_output

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
    
    def get_center_embedding(self, center):
        return self.center_embedding(center)

class word_embed_ng(nn.Module):
    def __init__(self, char_num, gen_embed_dim, hidden_size, num_layer, dropout, fc_hidden, embed_size, k, bidirectional, multigpu, device, models):
        super(word_embed_ng, self).__init__()
        self.center_generator = generator(char_num, gen_embed_dim, hidden_size, num_layer, dropout, bidirectional, multigpu, device)
        self.context_generator = generator(char_num, gen_embed_dim, hidden_size, num_layer, dropout, bidirectional, multigpu, device)
        self.mlp_center = nn.Linear(hidden_size, fc_hidden)
        self.mlp_context= nn.Linear(hidden_size, fc_hidden)
        self.k = k
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.last_fc_cen = nn.Linear(fc_hidden, embed_size)
        self.last_fc_con = nn.Linear(fc_hidden, embed_size)
        self.models = models

    def cal_loss(self, x, y, neg):
        score_target = torch.bmm(x.unsqueeze(1),y.unsqueeze(2))
        score_neg = torch.bmm(x.unsqueeze(1), neg.transpose(0,1).transpose(1,2))
        loss = -F.logsigmoid(score_target).sum() + -F.logsigmoid(-score_neg).sum()
        return loss

    def forward(self, x, x_len, y, y_len, neg):
        if self.models == "tanh":
            prediction = self.tanh(self.mlp_center(self.center_generator(x, x_len)))
            target = self.tanh(self.mlp_context(self.context_generator(y, y_len)))
            neg_output =[]
            for i in range(self.k):
                neg_output.append(self.tanh(self.mlp_context(self.context_generator(neg[i][0], neg[i][1]))))
        elif self.models == "linear":
            prediction = self.mlp_center(self.center_generator(x, x_len))
            target = self.mlp_context(self.context_generator(y, y_len))
            neg_output =[]
            for i in range(self.k):
                neg_output.append(self.mlp_context(self.context_generator(neg[i][0], neg[i][1])))
        else:
            prediction = self.last_fc_cen(self.tanh(self.mlp_center(self.center_generator(x, x_len))))
            target = self.last_fc_con(self.tanh(self.mlp_context(self.context_generator(y, y_len))))
            neg_output =[]
            for i in range(self.k):
                neg_output.append(self.last_fc_con(self.tanh(self.mlp_context(self.context_generator(neg[i][0], neg[i][1])))))
        neg_output_tensor = torch.stack(neg_output)
        loss = self.cal_loss(prediction, target, neg_output_tensor)
        return loss
    
    def get_center_embedding(self, center, center_len):
        if self.models == 'tanh':
            embedding = self.tanh(self.mlp_center(self.center_generator(center, center_len)))
        elif self.models == 'linear':
            embedding = self.mlp_center(self.center_generator(center, center_len))
        else:
            embedding = self.last_fc_cen(self.tanh(self.mlp_center(self.center_generator(center, center_len))))
        return embedding

class pretrained(nn.Module):
    def __init__(self, char_num, gen_embed_dim, hidden_size, num_layer, dropout, fc_hidden, embed_size, k, bidirectional, multigpu, device, models):
        super(pretrained, self).__init__()
        self.embedding_generator = generator(char_num, gen_embed_dim, hidden_size, num_layer, dropout, bidirectional, multigpu, device)
        self.mlp = nn.Linear(hidden_size, fc_hidden)
        self.tanh = nn.Tanh()
        self.last_fc = nn.Linear(fc_hidden, embed_size)
        self.models = models

    def cal_loss(self, predicted, target):
        loss = nn.MSELoss(size_average=False)
        return loss(predicted, target)

    def forward(self, word, word_len, true_embedding):
        if self.models == 'tanh':
            predicted_embedding = self.tanh(self.mlp(self.embedding_generator(word, word_len)))
        elif self.models == 'linear':
            predicted_embedding = self.mlp(self.embedding_generator(word, word_len))
        else:
            predicted_embedding = self.last_fc(self.tanh(self.mlp(self.embedding_generator(word, word_len))))

        loss = self.cal_loss(predicted_embedding, true_embedding)
        return loss

class pretrained_attn(nn.Module):
    def __init__(self, char_num, gen_embed_dim, hidden_size, num_layer, dropout, fc_hidden, embed_size, k, bidirectional, multigpu, device, models, attn_size):
        super(pretrained_attn, self).__init__()
        self.embedding_generator = generator(char_num, gen_embed_dim, hidden_size, num_layer, dropout, bidirectional, multigpu, device)
        self.mlp = nn.Linear(hidden_size, embed_size)
        # self.last_fc = nn.Linear(fc_hidden, embed_size)
        self.models = models
        self.hidden_size = hidden_size
        self.attn = nn.Sequential(
                        nn.Linear(hidden_size, attn_size),
                        nn.Tanh(),
                        nn.Linear(attn_size, 1)
        )
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size)

    def cal_loss(self, predicted, target):
        loss = nn.MSELoss(size_average=False)
        return loss(predicted, target)

    def forward(self, word, word_len, true_embedding):
        h, h_total = self.embedding_generator(word, word_len)
        b_size = h_total.size(0)
        attn = self.attn(h_total.view(-1, self.hidden_size))
        attn_weight = F.softmax(attn.view(b_size, -1), dim=1).unsqueeze(2)
        attn_hidden = (h_total*attn_weight).sum(dim=1)
        predicted_embedding = self.mlp(attn_hidden)
        loss = self.cal_loss(predicted_embedding, true_embedding)
        return loss

if __name__=='__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = word_embed_ng(26, 10, 10, 1, 0.3, 10, 5, False, True, device)
    # model = model.to(device)
    
    # text_loader = TextDataLoader('./data', batch_size = 2, window_size = 5, k=5)
    text_loader = TextDataLoader('./data', 'toy/merge.txt', 8, 5, 5, True, 0, 5, 1e-04)
    
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
            model = nn.DataParallel(model, device_ids=[2,3])
        model = model.to(device)
        output = model(center, center_len, context, context_len, neg)
        print(output)
        break
        
