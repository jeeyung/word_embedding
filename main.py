import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from model import *
import argparse
import torch.optim as optim
from datetime import datetime
from tensorboardX import SummaryWriter
import time

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=5, type=int,
                        help='number of total epochs (default: 32)')
    parser.add_argument('--lr', default=0.0002, type=float,
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--data-dir', default='data', type=str,
                        help='directory of training/testing data (default: datasets)')
    parser.add_argument('--timestamp', default=datetime.now().strftime("%y%m%d%H%M%S"), type=str)
    parser.add_argument('--load-model', action='store_true', default = False)
    parser.add_argument('--log-dir', default='saved/runs/', type=str)
    parser.add_argument('--vocab-size', default=27, type=int)
    parser.add_argument('--log_frequency', default=1000, type=int)
    parser.add_argument('--hidden-size', default=256, type=int)
    parser.add_argument('--window-size', default=5, type=int)
    parser.add_argument('--dropout', default=0.2, type= float)
    parser.add_argument('--embed-size', default=128, type=int)
    parser.add_argument('--num-layer', default=1, type=int)
    parser.add_argument('--mlp_size', default=128, type=int)
    parser.add_argument('--neg_sample_size', default=5, type=int)

    args = parser.parse_args()
    config_list = [args.batch_size, args.vocab_size, args.hidden_size, args.embed_size]
    config = ""
    for i in map(str, config_list):
        config = config + '_' + i
    args.config = config
    return args

def train(args):
    start_time = time.time()
    writer = SummaryWriter(args.log_dir + args.timestamp + args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model= word_embed_ng(args.vocab_size, args.embed_size, args.hidden_size,
                        args.num_layer, args.dropout, args.mlp_size, args.neg_sample_size)
    model= model.to(device)
    if args.load_model:
        model.load_state_dict(torch.load(args.log_dir + 'model_best.pt'))
        print('Model loaded')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    text_loader = TextDataLoader(args.data_dir, args.batch_size, args.window_size, args.neg_sample_size)
    train_loss = 0
    for epoch in range(args.epochs):
        monitor_loss = 0
        for i, (center, center_len, context, context_len, neg) in enumerate(text_loader):
            optimizer.zero_grad()
            loss = model(center, center_len, context, context_len, neg)
            loss.backward()
            optimizer.step()
            monitor_loss += loss.item()
            if i % args.log_frequency == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.8f}'.format(
                    (epoch + 1), i* args.batch_size, len(text_loader.dataset),
                    100. * i / len(text_loader),
                    loss/args.batch_size))
                step = epoch * len(text_loader) // args.log_frequency + i // args.log_frequency
                writer.add_scalar('Batch loss', loss / args.batch_size, step)

        print('====> Epoch: {} Average loss: {:.4f} / Time: {:.4f}'.format(
        (epoch+1),
        monitor_loss/ len(text_loader.dataset),
        time.time() - start_time))
        if train_loss > monitor_loss:
            torch.save(model.state_dict(), args.log_dir + 'model_best.pt')
        train_loss = monitor_loss
    writer.add_scalar('Train loss', train_loss / len(text_loader.dataset), (epoch+1))

if __name__ =='__main__':
    train(arg_parse())