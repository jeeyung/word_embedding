import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from model import *
from configuration import get_config
import torch.optim as optim
from tensorboardX import SummaryWriter
import time
from dataloader import TextDataLoader
import os
import math 
from evaluate import evaluate
from torch.optim.lr_scheduler import StepLR
from utils import result2dict

def train(args):
    
    datasetlist_dir = ["A","B","C","D","E","F","G","H","I","J","K","L"] 
    device = args.device
    if args.is_character:
        args.model_name = "cha-level"
    if args.model_name == 'sgns':
        # model = skipgram(len(text_loader.dataset.vocabs), args.embed_size)
        model = skipgram(40000, args.embed_size)
    else:
        model = word_embed_ng(args.vocab_size, args.embed_size, args.hidden_size,
                            args.num_layer, args.dropout, args.mlp_size, args.neg_sample_size, args.bidirectional, args.multigpu, args.device)
    # if torch.cuda.device_count() > 1 and args.multi:
    #     print("using", torch.cuda.device_count(), "GPUs")
    #     model = nn.DataParallel(model)
    model= model.to(device)
    print("made model")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    if args.load_model_code is not None:
        model.load_state_dict(torch.load(args.log_dir + args.load_model_code + '/model_best.pt'))
        args.timestamp = args.load_model_code[:12]
        print('Model loaded')
    for epoch in range(args.epochs):
        dataset_order = 0
        total_dataset_num = 0
        train_loss = 0
        monitor_loss = 0
        for dataset_dir in datasetlist_dir:
            for k in range(100):
                scheduler.step()
                start_time = time.time()
                wiki_datadir = args.dataset + dataset_dir
                dataset = os.path.join(wiki_datadir, 'wiki_{0:02d}.bz2'.format(k+args.dataset_order))
                text_loader = TextDataLoader(args.data_dir, dataset, args.batch_size, args.window_size, args.neg_sample_size,
                                        args.is_character, args.num_workers, args.remove_th, args.subsample_th)
                print("made text loader")
                writer = SummaryWriter(args.log_dir + args.timestamp + '_' + args.config)
                for i, (center,context, neg) in enumerate(text_loader):
                    if args.is_character:
                        center, center_len = center
                        context, context_len = context
                        center = center.to(device)
                        context = context.to(device)
                        optimizer.zero_grad()
                        loss = model(center, center_len, context, context_len, neg)
                    else:
                        center = center.to(device)
                        context = context.to(device)
                        neg = neg.to(device)
                        optimizer.zero_grad()
                        loss = model(center, context, neg)
                    loss.backward()
                    optimizer.step()
                    monitor_loss += loss.item()
                    if i % args.log_frequency == 0:
                        print('Train dataset: {} [{}/{} ({:.0f}%)] Loss: {:.8f}'.format(
                            (dataset_order), i* args.batch_size, len(text_loader.dataset),
                            100. * i / len(text_loader),
                            loss/args.batch_size))
                        step = i // args.log_frequency + math.ceil(total_dataset_num // args.batch_size // args.log_frequency)
                        writer.add_scalar('Batch loss', loss / args.batch_size, step)
                dataset_order += 1
                total_dataset_num += len(text_loader.dataset)
                print('====> Dataset: {} Average loss: {:.4f} / Time: {:.4f}'.format(
                dataset_order,
                monitor_loss/ total_dataset_num,
                time.time() - start_time))
                if k % args.save_frequency ==0:
                    torch.save(model.state_dict(), args.log_dir + args.timestamp + '_' + args.config + '/' +f'model_{k}.pt')
                    print("Model saved")
                if train_loss > monitor_loss/total_dataset_num:
                    torch.save(model.state_dict(), args.log_dir + args.timestamp + '_' + args.config + '/model_best.pt')
                    print("Model saved")
                train_loss = monitor_loss/total_dataset_num
                writer.add_scalar('Train loss', monitor_loss/total_dataset_num, dataset_order)
                if args.model_name == "sgns":
                    sim_results = evaluate(model.state_dict(), text_loader.dataset.word2idx, True)
                    ana_results = evaluate(model.state_dict(), text_loader.dataset.word2idx, False)
                    sim_score, sim_known = result2dict(sim_results)
                    ana_score, ana_known = result2dict(ana_results)
                    writer.add_scalars('Similarity score', sim_score, k)
                    writer.add_scalars('Similarity known', sim_known, k)
                    writer.add_scalars('Analogy score', ana_score, k)
                    writer.add_scalars('Analogy known', ana_known, k)
                writer.add_scalar('Epoch time', time.time() - start_time, k)
                del text_loader

if __name__ =='__main__':
    train(get_config())