import argparse
import torch
from datetime import datetime
from pathlib import Path
import re
home = str(Path.home())


def get_config():
    parser = argparse.ArgumentParser()

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--model-name', default='sgns', type=str)
    model_arg.add_argument('--model-category', default='fc_acti', type=str)
    model_arg.add_argument('--embed-size', default=300, type=int)
    model_arg.add_argument('--char-embed-size', default=128, type=int)
    model_arg.add_argument('--vocab-size', default=28, type=int)
    model_arg.add_argument('--hidden-size', default=512, type=int)
    model_arg.add_argument('--dropout', default=0.2, type= float)
    model_arg.add_argument('--num-layer', default=1, type=int)
    model_arg.add_argument('--mlp-size', default=400, type=int)
    model_arg.add_argument('--bidirectional', action='store_true')
    model_arg.add_argument('--attn-size', default=300, type=int)

    data_arg = parser.add_argument_group('Data')
    data_arg.add_argument('--data-dir', default='data', type=str, help='directory of training/testing data (default: datasets)')
    # data_arg.add_argument('--data-dir', default='/data/jeeyung', type=str, help='directory of training/testing data (default: datasets)')
    data_arg.add_argument('--dataset', default='toy/merge.txt', type=str)
    # data_arg.add_argument('--dataset', default='wiki_dump/', type=str)
    data_arg.add_argument('--dataset-f-name', default=None, type=str)
    data_arg.add_argument('--window-size', default=5, type=int)
    data_arg.add_argument('--neg-sample-size', default=7, type=int)
    data_arg.add_argument('--is-character', action='store_true')
    data_arg.add_argument('--dataset-order', default=0, type=int)
    data_arg.add_argument('--remove-th', default=3, type=int)
    data_arg.add_argument('--subsample-th', default=1e-4, type=float)
 
    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--device', default=0, type=int)
    train_arg.add_argument('--batch-size', default=64, type=int, help='mini-batch size (default: 64)')
    train_arg.add_argument('--epochs', default=128, type=int, help='number of total epochs (default: 10)')
    train_arg.add_argument('--lr', default=0.025, type=float, help='learning rate (default: 0.0002)')
    train_arg.add_argument('--clip', default=0.25, type=float)
    train_arg.add_argument('--log-frequency', default=100, type=int)
    train_arg.add_argument('--save-frequency', default=2, type=int)
    train_arg.add_argument('--timestamp', default=datetime.now().strftime("%y%m%d%H%M%S"), type=str)
    train_arg.add_argument('--load-model', default=None, type=str)
    train_arg.add_argument('--load-model-code', default=None, type=str)
    train_arg.add_argument('--load-best-model', action='store_true')
    train_arg.add_argument('--log-dir', default='saved/runs/', type=str)
    train_arg.add_argument('--multigpu', action='store_true')
    train_arg.add_argument('--is_ngram', action='store_true')
    train_arg.add_argument('--is_attn', action='store_true')
    train_arg.add_argument('--load_pretrained', action='store_true')
    train_arg.add_argument('--load-pretrained-code', default=None, type=str)

    train_arg.add_argument('--evaluation', action='store_true')
    #for large dataset dataloader
    train_arg.add_argument('--num-workers', default=0, type=int)
    # test_arg = parser.add_argument_group('Test')
    # test_arg.add_argument('--load-model', action='store_true', default = False)
    train_arg.add_argument('--multi-node', action='store_true')
    train_arg.add_argument('--memo', default='', type=str)
    train_arg.add_argument('--backend', default='nccl', type=str)
    train_arg.add_argument('--init-method', default='nccl://127.0.0.1:22', type=str)
    train_arg.add_argument('--rank', type=int)
    train_arg.add_argument('--world-size', type=int)

    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.is_character:
        args.model_name = 'lstm'
    if args.is_ngram:
        args.vocab_size = 730
    if args.is_character and args.model_category is None:
        parser.error('model category is required when is-character is True')

    config_list = [args.model_name, args.embed_size, args.hidden_size,\
                   args.dataset, args.window_size, args.neg_sample_size, args.is_character,\
                   args.device, args.batch_size, args.epochs, args.lr, args.bidirectional, args.num_layer, args.model_category,
                   args.is_attn, args.is_ngram,  args.memo]
    args.config = '_'.join(list(map(str, config_list))).replace("/", ".")
    return args
