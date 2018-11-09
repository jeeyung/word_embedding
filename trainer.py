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
from evaluate import evaluate

def train(args):
    start_time = time.time()
    device = args.device
    text_loader = TextDataLoader(args.data_dir, args.dataset, args.batch_size, args.window_size, args.neg_sample_size,
                                 args.is_character, args.num_workers)
    if args.is_character:
        args.model_name = "cha-level"
    if args.model_name == 'sgns':
        model = skipgram(len(text_loader.dataset.vocabs), args.embed_size)
    else:
        model = word_embed_ng(args.vocab_size, args.embed_size, args.hidden_size,
                            args.num_layer, args.dropout, args.mlp_size, args.neg_sample_size)
    model= model.to(device)
    if args.load_model:
        model.load_state_dict(torch.load(args.log_dir + args.load_model_code + '/model_best.pt'))
        args.timestamp = args.load_model_code[:12]
        print('Model loaded')

    writer = SummaryWriter(args.log_dir + args.timestamp + '_' + args.config)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loss = 0
    for epoch in range(args.epochs):
        monitor_loss = 0
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
            torch.save(model.state_dict(), args.log_dir + args.timestamp + '_' + args.config + '/model_best.pt')
            print("Model saved")
        train_loss = monitor_loss
        writer.add_scalar('Train loss', train_loss / len(text_loader.dataset), (epoch+1))
        evaluate(model.state_dict(), text_loader.dataset.word2idx, True)

if __name__ =='__main__':
    train(get_config())