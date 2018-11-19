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
import csv

def train_epoch(args, model, device, epoch, monitor_loss, optimizer, scheduler, writer, text_loader, dataset_order, total_dataset_num):
    scheduler.step()
    for i, (center,context, neg) in enumerate(text_loader):
        if args.is_character:
            center, center_len = center
            context, context_len = context
            center = center.to(device)
            context = context.to(device)
            n=[]
            for k in range(args.neg_sample_size):
                padded_neg, neg_len = neg[k]
                n.append((padded_neg.to(device), neg_len))
            optimizer.zero_grad()
            loss = model(center, center_len, context, context_len, n)
        else:
            center = center.to(device)
            context = context.to(device)
            neg = neg.to(device)
            optimizer.zero_grad()
            loss = model(center, context, neg)
        loss.backward()
        if not args.model_name == 'sgns':
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        monitor_loss += loss.item()
        if i % args.log_frequency == 0:
            print('Train dataset: {} [{}/{} ({:.0f}%)] Loss: {:.8f}'.format(
                (dataset_order), i* args.batch_size, len(text_loader.dataset),
                100. * i / len(text_loader),
                loss/args.batch_size))
            if args.dataset == "wiki_dump/":
                step = i // args.log_frequency + math.ceil(total_dataset_num // args.batch_size // args.log_frequency)
            else:
                step = i // args.log_frequency + epoch * len(text_loader) // args.log_frequency
            writer.add_scalar('Batch loss', loss / args.batch_size, step)
            # plot_embedding(args, model, text_loader, writer, device)
    if args.evaluation:
        if args.dataset =="wiki_dump/":
            evaluation(args, writer, model, device, text_loader, dataset_order)
        else:
            evaluation(args, writer, model, device, text_loader, epoch)
    return monitor_loss

def evaluation(args, writer, model, device, text_loader, k):
    if args.model_name == "sgns":
        sim_results = evaluate(model, device, True, text_loader.dataset.word2idx)
        ana_results = evaluate(model, device, False, text_loader.dataset.word2idx)
    else:
        sim_results = evaluate(model, device, True)
        ana_results = evaluate(model, device, False)
    sim_score, sim_known = result2dict(sim_results)
    ana_score, ana_known = result2dict(ana_results)
    writer.add_scalars('Similarity score', sim_score, k)
    writer.add_scalars('Similarity known', sim_known, k)
    writer.add_scalars('Analogy score', ana_score, k)
    writer.add_scalars('Analogy known', ana_known, k)

def plot_embedding(args, model, text_loader, device, epoch, writer):
    writer = SummaryWriter(args.log_dir + args.timestamp + '_' + args.config + '/' + str(epoch))
    vocabs = text_loader.dataset.vocabs
    if args.model_name == 'sgns':
        tokenized = [text_loader.dataset.word2idx[vocab] for vocab in vocabs]
        tokenized= torch.LongTensor(tokenized)
        features = model.get_center_embedding(tokenized.to(device))
    else:
        tokenized = [[text_loader.dataset.char2idx[character] for character in vocab] for vocab in vocabs]
        tokenized.sort(key=lambda x: len(x), reverse=True)
        token_lengths = list(map(len, tokenized))
        token_tensor = torch.zeros(len(tokenized), max(token_lengths), dtype=torch.long)
        for idx, (token, tokenlen) in enumerate(zip(tokenized, token_lengths)):
            token_tensor[idx, :tokenlen] = torch.LongTensor(token)
        features = model.get_center_embedding(token_tensor.to(device), token_lengths)
    writer.add_embedding(features, metadata=vocabs)
    print("plot embedding")

def train(args):
    device = args.device
    if args.dataset == "wiki_dump/":
        if args.dataset_f_name == "B":
            datasetlist_dir = ["B","C","D","E","F","G","H","I","J","K","L"] 
        elif args.dataset_f_name == "C":
            datasetlist_dir = ["C","D","E","F","G","H","I","J","K","L"] 
        else:
            datasetlist_dir = ["A","B","C","D","E","F","G","H","I","J","K","L"] 
        if args.model_name == 'sgns':
            model = skipgram(40000, args.embed_size)
        else:
            model = word_embed_ng(args.vocab_size, args.char_embed_size, args.hidden_size,
                                args.num_layer, args.dropout, args.mlp_size, args.embed_size, args.neg_sample_size, args.bidirectional,
                                args.multigpu, args.device, args.model_category)
    else:
        text_loader = TextDataLoader(args.data_dir, args.dataset, args.batch_size, args.window_size, args.neg_sample_size,
                                 args.is_character, args.num_workers, args.remove_th, args.subsample_th)
        if args.model_name == 'sgns':
            # model = skipgram(len(text_loader.dataset.vocabs), args.embed_size)
            model = skipgram(40000, args.embed_size)
        else:
            model = word_embed_ng(args.vocab_size, args.char_embed_size, args.hidden_size,
                                args.num_layer, args.dropout, args.mlp_size, args.embed_size, 
                                args.neg_sample_size, args.bidirectional, args.multigpu, args.device, args.model_category)
    if torch.cuda.device_count() > 1 and args.multigpu:
        print("using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model= model.to(device)
    print("made model")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    if args.load_model_code is not None:
        if args.load_best_model:
            model_name = '/model_best.pt'
        else:
            model_name = '/model.pt'
        # model.load_state_dict(torch.load(args.log_dir + args.load_model_code + model_name, map_location=lambda storage,loc: storage))  
        checkpoint = torch.load(args.log_dir + args.load_model_code + model_name, map_location=lambda storage,loc: storage)
        args.dataset_order += checkpoint['dataset_order']
        model.load_state_dict(checkpoint['model_state_dict'])
        args.timestamp = args.load_model_code[:12]
        print('Model loaded')
    writer = SummaryWriter(args.log_dir + args.timestamp + '_' + args.config)
    train_loss = 0
    for epoch in range(args.epochs):
        dataset_order = 0
        dataset_order += args.dataset_order
        total_dataset_num = 0
        monitor_loss = 0
        if args.dataset=="wiki_dump/":
            for dataset_dir in datasetlist_dir:
                for k in range(100):
                    if k >= 100:
                        break
                    start_time = time.time()
                    wiki_datadir = args.dataset + dataset_dir
                    dataset = os.path.join(wiki_datadir, 'wiki_{0:02d}.bz2'.format(k+args.dataset_order))
                    text_loader = TextDataLoader(args.data_dir, dataset, args.batch_size, args.window_size, args.neg_sample_size,
                                            args.is_character, args.num_workers, args.remove_th, args.subsample_th)
                    print("made text loader")
                    monitor_loss = train_epoch(args, model, device, epoch, monitor_loss, optimizer, scheduler, writer,
                                                    text_loader, dataset_order, total_dataset_num)
                    dataset_order += 1
                    total_dataset_num += len(text_loader.dataset)
                    print('====> Dataset: {} Average loss: {:.4f} / Time: {:.4f}'.format(
                    dataset_order,
                    monitor_loss/ total_dataset_num,
                    time.time() - start_time))
                    checkpoint = {'dataset_order':dataset_order, 'model_state_dict':model.state_dict()}
                    torch.save(checkpoint, args.log_dir + args.timestamp + '_' + args.config + '/' +'model.pt')
                    print("Model saved")
                    if train_loss > monitor_loss/total_dataset_num:
                        torch.save(model.state_dict(), args.log_dir + args.timestamp + '_' + args.config + '/model_best.pt')
                        print("Model saved")
                    train_loss = monitor_loss/total_dataset_num
                    writer.add_scalar('Train loss', monitor_loss/total_dataset_num, dataset_order)
                    writer.add_scalar('Epoch time', time.time() - start_time, k)
                    del text_loader
        else:
            monitor_loss = 0
            start_time = time.time()
            monitor_loss = train_epoch(args, model, device, epoch, monitor_loss, optimizer, scheduler,
                                            writer, text_loader, dataset_order, total_dataset_num)
            print('====> Epoch: {} Average loss: {:.4f} / Time: {:.4f}'.format(
                 (epoch), monitor_loss/ len(text_loader.dataset), time.time() - start_time))
            plot_embedding(args, model, text_loader, device, epoch, writer)
            torch.save(model.state_dict(), args.log_dir + args.timestamp + '_' + args.config + '/' +'model.pt')
            print("Model saved")
            if train_loss > monitor_loss:
                torch.save(model.state_dict(), args.log_dir + args.timestamp + '_' + args.config + '/model_best.pt')
                print("Best model saved")
            writer.add_scalar('Epoch time', time.time() - start_time, epoch)
            train_loss = monitor_loss

if __name__ =='__main__':
    train(get_config())