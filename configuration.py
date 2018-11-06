import argparse
import torch
from datetime import datetime
from pathlib import Path
home = str(Path.home())


def get_config():
    parser = argparse.ArgumentParser()

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--model-name', default='sgns', type=str)
    model_arg.add_argument('--embed-size', default=128, type=int)
    model_arg.add_argument('--vocab-size', default=27, type=int)
    model_arg.add_argument('--hidden-size', default=256, type=int)
    model_arg.add_argument('--dropout', default=0.2, type= float)
    model_arg.add_argument('--num-layer', default=1, type=int)
    model_arg.add_argument('--mlp_size', default=128, type=int)

    data_arg = parser.add_argument_group('Data')
    data_arg.add_argument('--data-dir', default='data', type=str, help='directory of training/testing data (default: datasets)')
    data_arg.add_argument('--dataset', default='toy/merge.txt', type=str)
    data_arg.add_argument('--window-size', default=5, type=int)
    data_arg.add_argument('--neg-sample-size', default=5, type=int)
    data_arg.add_argument('--is-character', action='store_true')

    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--device', default=0, type=int)
    train_arg.add_argument('--batch-size', default=32, type=int, help='mini-batch size (default: 32)')
    train_arg.add_argument('--epochs', default=10, type=int, help='number of total epochs (default: 10)')
    train_arg.add_argument('--lr', default=0.0002, type=float, help='learning rate (default: 0.0002)')
    train_arg.add_argument('--log_frequency', default=1000, type=int)
    train_arg.add_argument('--timestamp', default=datetime.now().strftime("%y%m%d%H%M%S"), type=str)
    train_arg.add_argument('--load-model', action='store_false')
    train_arg.add_argument('--log-dir', default='saved/runs/', type=str)

    # test_arg = parser.add_argument_group('Test')
    # test_arg.add_argument('--load-model', action='store_true', default = False)
    
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    config_list = [args.model_name, args.embed_size,\
                   args.dataset, args.window_size, args.neg_sample_size, args.is_character,\
                   args.device, args.batch_size, args.epochs, args.lr]
    args.config = '_'.join(list(map(str, config_list)))
    return args
