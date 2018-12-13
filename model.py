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
        self.bidirectional = bidirectional
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
        output, (h,_) = self.lstm(embedded)
        if self.bidirectional:
            ordered_hidden_1 = h[-1].index_select(0, unsort_idx)
            ordered_hidden_2 = h[-2].index_select(0, unsort_idx)
            ordered_hidden = torch.cat((ordered_hidden_1,ordered_hidden_2), dim=1)
            output_padded, _ = pad_packed_sequence(output, batch_first=True)
            ordered_output = output_padded.index_select(0, unsort_idx)
        else:
            ordered_hidden = h[-1].index_select(0, unsort_idx)
            output_padded, _ = pad_packed_sequence(output, batch_first=True)
            ordered_output = output_padded.index_select(0, unsort_idx)
        return ordered_hidden, ordered_output

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
    def __init__(self, char_num, gen_embed_dim, hidden_size, num_layer, dropout, fc_hidden, embed_size, k, bidirectional, multigpu, device, models, is_attn):
        super(word_embed_ng, self).__init__()
        self.center_generator = generator(char_num, gen_embed_dim, hidden_size, num_layer, dropout, bidirectional, multigpu, device)
        self.context_generator = generator(char_num, gen_embed_dim, hidden_size, num_layer, dropout, bidirectional, multigpu, device)
        self.is_attn = is_attn
        self.k = k
        self.hidden_size = hidden_size
        self.model_name = models
        self.cen_add_fc= nn.Sequential(
            nn.Linear(hidden_size, embed_size)
        )
        self.con_add_fc= nn.Sequential(
            nn.Linear(hidden_size, embed_size)
        )
        self.cen_add_fc_activation= nn.Sequential(
            nn.Linear(hidden_size, embed_size),
            nn.Tanh()
        )
        self.con_add_fc_activation= nn.Sequential(
            nn.Linear(hidden_size, embed_size),
            nn.Tanh()
        )
        self.cen_add_mlp= nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.Tanh(),
            nn.Linear(fc_hidden, embed_size)
        )
        self.con_add_mlp = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.Tanh(),
            nn.Linear(fc_hidden, embed_size)
        )
        self.cen_attn = nn.Sequential(
                        nn.Linear(self.hidden_size, attn_size),
                        nn.Tanh(),
                        nn.Linear(attn_size, 1)
        )
        self.con_attn = nn.Sequential(
                        nn.Linear(self.hidden_size, attn_size),
                        nn.Tanh(),
                        nn.Linear(attn_size, 1)
        )

    def cal_loss(self, x, y, neg):
        score_target = torch.bmm(x.unsqueeze(1),y.unsqueeze(2))
        score_neg = torch.bmm(x.unsqueeze(1), neg.transpose(0,1).transpose(1,2))
        loss = -F.logsigmoid(score_target).sum() + -F.logsigmoid(-score_neg).sum()
        return loss

    def forward(self, x, x_len, y, y_len, neg):
        _, cen_output = self.center_generator(x, x_len)
        # embedded_con, con_output = self.context_generator(y, y_len)
        if self.is_attn:
            b_size = cen_output.size(0)
            attn = self.cen_attn(cen_output.view(-1, self.hidden_size))
            attn_weight = F.softmax(attn.view(b_size, -1), dim=1).unsqueeze(2)
            embedded_cen = (cen_output*attn_weight).sum(dim=1)
        if self.model_name == "fc_acti":
            prediction = self.add_fc_activation_cen(embedded_cen)
            target = self.add_fc_activation_con(embedded_con)
            neg_output =[]
            for i in range(self.k):
                if self.is_attn:
                    _, neg_output = self.context_generator(neg[i][0], neg[i][1])
                    attn = self.con_attn(neg_output.view(-1, self.hidden_size))
                    attn_weight = F.softmax(attn.view(b_size, -1), dim=1).unsqueeze(2)
                    embedded_neg = (neg_output*attn_weight).sum(dim=1)
                    neg_output.append(self.add_fc_activation_con(embedded_neg))
        elif self.model_name == "fc":
            prediction = self.add_fc_cen(embedded_cen)
            target = self.add_fc_con(embedded_con)
            neg_output =[]
            for i in range(self.k):
                if self.is_attn:
                    _, neg_output = self.context_generator(neg[i][0], neg[i][1])
                    attn = self.con_attn(neg_output.view(-1, self.hidden_size))
                    attn_weight = F.softmax(attn.view(b_size, -1), dim=1).unsqueeze(2)
                    embedded_neg = (neg_output*attn_weight).sum(dim=1)
                    neg_output.append(self.add_fc_activation_con(embedded_neg))
        else:
            prediction = self.add_mlp_cen(embedded_cen)
            target = self.add_mlp_con(embedded_con)
            neg_output =[]
            for i in range(self.k):
                if self.is_attn:
                    _, neg_output = self.context_generator(neg[i][0], neg[i][1])
                    attn = self.con_attn(neg_output.view(-1, self.hidden_size))
                    attn_weight = F.softmax(attn.view(b_size, -1), dim=1).unsqueeze(2)
                    embedded_neg = (neg_output*attn_weight).sum(dim=1)
                    neg_output.append(self.add_fc_activation_con(embedded_neg))
        neg_output_tensor = torch.stack(neg_output)
        loss = self.cal_loss(prediction, target, neg_output_tensor)
        return loss
    
    def get_center_embedding(self, center, center_len):
        embedded_cen, _ = self.center_generator(center, center_len)
        if self.model_name == "fc_acti":
            embedding = self.add_fc_activation_cen(embedded_cen)
        elif self.model_name == "fc":
            embedding = self.add_fc_cen(embedded_cen)
        else:
            embedding = self.add_mlp_cen(embedded_cen)
        return embedding

class pretrained(nn.Module):
    def __init__(self, char_num, gen_embed_dim, hidden_size, num_layer, dropout, 
                fc_hidden, embed_size, k, bidirectional, multigpu, device, models, is_attn, attn_size):
        super(pretrained, self).__init__()
        self.embedding_generator = generator(char_num, gen_embed_dim, hidden_size, num_layer, dropout, bidirectional, multigpu, device)
        self.model_name = models
        self.is_attn = is_attn
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.hidden_size = hidden_size*2 
        self.add_fc= nn.Sequential(
            nn.Linear(self.hidden_size, embed_size)
        )
        self.add_fc_activation= nn.Sequential(
            nn.Linear(self.hidden_size, embed_size),
            nn.Tanh()
        )
        self.add_mlp= nn.Sequential(
            nn.Linear(self.hidden_size, fc_hidden),
            nn.Tanh(),
            nn.Linear(fc_hidden, embed_size)
        )
        self.attn = nn.Sequential(
                        nn.Linear(self.hidden_size, attn_size),
                        nn.Tanh(),
                        nn.Linear(attn_size, 1)
        )

    def cal_loss(self, predicted, target):
        # loss = nn.PairwiseDistance()
        loss = nn.CosineSimilarity()
        # loss = nn.MSELoss(size_average=False)
        return loss(predicted, target)

    def forward(self, word, word_len, true_embedding):
        hidden, output = self.embedding_generator(word, word_len)
        if self.is_attn:
            b_size = output.size(0)
            attn = self.attn(output.view(-1, self.hidden_size))
            attn_weight = F.softmax(attn.view(b_size, -1), dim=1).unsqueeze(2)
            hidden = (output*attn_weight).sum(dim=1)
        if self.model_name == 'fc_acti':
            predicted_embedding = self.add_fc_activation(hidden)
        elif self.model_name == 'fc':
            predicted_embedding = self.add_fc(hidden)
        else:
            predicted_embedding = self.add_mlp(hidden)
        # for pairwisedistance
        # loss = self.cal_loss(predicted_embedding, true_embedding).sum()
        # for cosine similarity
        loss = -self.cal_loss(predicted_embedding, true_embedding).sum()
        # for mse
        # loss = self.cal_loss(predicted_embedding, true_embedding)
        return loss

class pretrained_test(nn.Module):
    def __init__(self, char_num, gen_embed_dim, hidden_size, num_layer, dropout, 
                fc_hidden, embed_size, k, bidirectional, multigpu, device, models, is_attn, attn_size):
        super(pretrained_test, self).__init__()
        self.embedding_generator = generator(char_num, gen_embed_dim, hidden_size, num_layer, dropout, bidirectional, multigpu, device)
        self.model_name = models
        self.is_attn = is_attn
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.hidden_size = hidden_size*2 
        self.add_fc= nn.Sequential(
            nn.Linear(self.hidden_size, embed_size)
        )
        self.add_fc_activation= nn.Sequential(
            nn.Linear(self.hidden_size, embed_size),
            nn.Tanh()
        )
        self.add_mlp= nn.Sequential(
            nn.Linear(self.hidden_size, fc_hidden),
            nn.Tanh(),
            nn.Linear(fc_hidden, embed_size)
        )
        self.attn = nn.Sequential(
                        nn.Linear(self.hidden_size, attn_size),
                        nn.Tanh(),
                        nn.Linear(attn_size, 1)
        )

    def cal_loss(self, predicted, target):
        inner_pre = torch.matmul(predicted, predicted.t())
        inner_tar = torch.matmul(target, target.t())
        loss = nn.MSELoss(size_average=False)
        return loss(inner_pre, inner_tar)

    def forward(self, word, word_len, true_embedding):
        hidden, output = self.embedding_generator(word, word_len)
        if self.is_attn:
            b_size = output.size(0)
            attn = self.attn(output.view(-1, self.hidden_size))
            attn_weight = F.softmax(attn.view(b_size, -1), dim=1).unsqueeze(2)
            hidden = (output*attn_weight).sum(dim=1)
        if self.model_name == 'fc_acti':
            predicted_embedding = self.add_fc_activation(hidden)
        elif self.model_name == 'fc':
            predicted_embedding = self.add_fc(hidden)
        else:
            predicted_embedding = self.add_mlp(hidden)
        # for pairwisedistance
        # loss = self.cal_loss(predicted_embedding, true_embedding).sum()
        # for cosine similarity
        #loss = -self.cal_loss(predicted_embedding, true_embedding).sum()
        # for mse
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
        
