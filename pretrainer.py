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
from torch import distributed, nn
from torch.utils.data.distributed import DistributedSampler
from trainer import Trainer

class Pretrainer(Trainer):
    def __init__(self, args, model, device, optimizer,
                        scheduler, writer, text_loader, epoch, monitor_loss, dataset_order, total_dataset_num):
        self.args = args
        self.model = model
        self.device = device
        self.epoch = epoch
        self.monitor_loss = monitor_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.text_loader = text_loader
        self.dataset_order = dataset_order
        self.total_dataset_num = total_dataset_num

    def train_epoch(self):
        self.scheduler.step()
        assert type(self.model).__name__ == "pretrained"
        for i, (word, word_len, embedding) in enumerate(self.text_loader):
            word = word.to(self.device)
            embedding = embedding.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(word, word_len, embedding)
            loss.backward()

            if not self.args.model_name == 'sgns':
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if self.args.multi_node:
                self.average_gradients()
                print('average gradient')

            self.optimizer.step()
            self.monitor_loss += loss.item()
            if i % self.args.log_frequency == 0:
                print('Train dataset: {} [{}/{} ({:.0f}%)] Loss: {:.8f}'.format(
                    (self.dataset_order), i * self.args.batch_size, len(self.text_loader.dataset),
                                          100. * i / len(self.text_loader),
                                          loss / self.args.batch_size))
                if self.args.dataset == "wiki_dump/":
                    step = i // self.args.log_frequency + math.ceil(
                        self.total_dataset_num // self.args.batch_size // self.args.log_frequency)
                else:
                    step = i // self.args.log_frequency + self.epoch * len(self.text_loader) // self.args.log_frequency
                self.writer.add_scalar('Batch loss', loss / self.args.batch_size, step)
                # plot_embedding(args, model, text_loader, writer, device)
        if self.args.evaluation:
            pass
        return self.monitor_loss


def train(args):
    device = args.device
    text_loader = PretrainedDataLoader(args.data_dir, args.batch_size)
    # TODO : make pretrained model class in model.py
    model = pretrained()

    model= model.to(device)
    print("made model")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    if args.load_model_code is not None:
        pass
        #if args.load_best_model:
        #    model_name = '/model_best.pt'
        #else:
        #    model_name = '/model.pt'
        # model.load_state_dict(torch.load(args.log_dir + args.load_model_code + model_name, map_location=lambda storage,loc: storage))
        #checkpoint = torch.load(args.log_dir + args.load_model_code + model_name, map_location=lambda storage,loc: storage)
        #args.dataset_order += checkpoint['dataset_order']
        #model.load_state_dict(checkpoint['model_state_dict'])
        #args.timestamp = args.load_model_code[:12]
        #print('Model loaded')

    writer = SummaryWriter(args.log_dir + args.timestamp + '_' + args.config)
    train_loss = 0
    trainer = Pretrainer(args, model, device, optimizer, scheduler, writer,
                         text_loader, epoch=0, monitor_loss=0, dataset_order=0, total_dataset_num=0)

    for epoch in range(args.epochs):
        trainer.monitor_loss = 0
        trainer.epoch = epoch

        start_time = time.time()
        monitor_loss = trainer.train_epoch()
        print('====> Epoch: {} Average loss: {:.4f} / Time: {:.4f}'.format(
            (epoch), monitor_loss / len(text_loader.dataset), time.time() - start_time))
        #if epoch % 10 == 0:
        #    plot_embedding(args, model, text_loader, device, epoch, writer)
        torch.save(model.state_dict(), args.log_dir + args.timestamp + '_' + args.config + '/' +'model.pt')
        print("Model saved")
        if train_loss > monitor_loss:
            torch.save(model.state_dict(), args.log_dir + args.timestamp + '_' + args.config + '/model_best.pt')
            print("Best model saved")
        writer.add_scalar('Epoch time', time.time() - start_time, epoch)
        train_loss = monitor_loss


if __name__ == "__main__":
    train(get_config())