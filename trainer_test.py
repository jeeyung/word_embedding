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

def train(args):
    start_time = time.time()
    writer = SummaryWriter(args.log_dir + args.timestamp + args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_loader = TextDataLoader(args.data_dir, args.dataset, args.batch_size, args.window_size, args.neg_sample_size,
                                 args.is_character)
    if args.model_name == 'sgns':
        model = skipgram(len(text_loader.dataset.vocabs), args.embed_size)
    else:
        model = word_embed_ng(args.vocab_size, args.embed_size, args.hidden_size,
                            args.num_layer, args.dropout, args.mlp_size, args.neg_sample_size)
    model= model.to(device)
    if args.load_model:
        model.load_state_dict(torch.load(args.log_dir + 'model_best.pt'))
        print('Model loaded')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loss = 0



    data_list = os.
    for dataset in data_list:
        text_loader = TextDataLoader(args.data_dir, , args.batch)
        for epoch in range(args.epochs):
            monitor_loss = 0
            for i, (center,context, neg) in enumerate(text_loader):
                if args.is_character:
                    center, center_len = center[0]
                    context, context_len = context
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
                torch.save(model.state_dict(), args.log_dir + 'model_best.pt')
            train_loss = monitor_loss
        writer.add_scalar('Train loss', train_loss / len(text_loader.dataset), (epoch+1))

if __name__ =='__main__':
    train(get_config())