'''
MIT License

Copyright (c) 2017 Mat Leonard

'''

import argparse
import os

import numpy as np

from model import CharRNN, save_model, load_model, train
from utils import get_lookup_tables

parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_file', type=str,
                    help='input text file')
parser.add_argument('--save_dir', type=str, default='save',
                    help='directory to store checkpointed models')
parser.add_argument('--rnn_size', type=int, default=128,
                    help='size of RNN hidden state')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers in the RNN')
parser.add_argument('--batch_size', type=int, default=50,
                    help='minibatch size')
parser.add_argument('--seq_length', type=int, default=50,
                    help='RNN sequence length')
parser.add_argument('--num_epochs', type=int, default=25,
                    help='number of epochs')
parser.add_argument('--print_every', type=int, default=20,
                    help='print frequency')
parser.add_argument('--grad_clip', type=float, default=5.,
                    help='clip gradients at this value')
parser.add_argument('--learning_rate', type=float, default=0.002,
                    help='learning rate')
parser.add_argument('--dropout_prob', type=float, default=0.5,
                    help='probability of dropping weights')
parser.add_argument('--gpu', action='store_true', default=False,
                    help='run the network on the GPU')
parser.add_argument('--init_from', type=str, default=None,
                    help='initialize network from checkpoint')

args = parser.parse_args()

if not os.path.isdir(args.save_dir):
    raise OSError(f'Directory {args.save_dir} does not exist.')

with open(args.in_file, 'r') as f:
    text = f.read()

int2char, char2int = get_lookup_tables(text)
encoded = np.array([char2int[ch] for ch in text])
chars = tuple(char2int.keys())

if args.init_from is None:
    net = CharRNN(chars, n_hidden=args.rnn_size, n_layers=args.num_layers)
else:
    net = load_model(args.init_from)

val_loss = train(net, encoded, epochs=args.num_epochs, n_seqs=args.batch_size, 
                               n_steps=args.seq_length, lr=args.learning_rate, 
                               cuda=args.gpu, print_every=args.print_every)

save_file = f'charRNN_{val_loss:.4f}.ckpt'

save_model(net, os.path.join(args.save_dir, save_file))
print(f'Network saved as {save_file}')