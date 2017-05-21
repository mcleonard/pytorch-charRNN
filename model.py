'''
MIT License

Copyright (c) 2017 Mat Leonard

'''

import torch.nn as torch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class CharRNN(nn.Module):
    def __init__(self, tokens, n_steps=100, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        #self.embed = nn.Embedding(len(self.chars), embed_dim)
        self.dropout = nn.Dropout(drop_prob)
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(n_hidden, len(self.chars))
        
        self.init_weights()
        
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x, hc):
        #x = self.embed(x)
        #x = self.dropout(x)

        x, (h, c) = self.lstm(x, hc)
        x = self.dropout(x)
        
        # Stack up LSTM outputs
        x = x.view(x.size()[0]*x.size()[1], self.n_hidden)
        
        x = self.fc(x)
        
        return x, (h, c)
    
    def predict(self, char, h=None, cuda=False, top_k=None):
        ''' Given a character, predict the next character.
        
            Returns the predicted character and the hidden state.
        '''
        if cuda:
            self.cuda()
        else:
            self.cpu()
        
        if h is None:
            h = self.init_hidden(1)
        
        x = np.array([[self.char2int[char]]])
        x = one_hot_encode(x, len(self.chars))
        inputs = Variable(torch.from_numpy(x), volatile=True)
        if cuda:
            inputs = inputs.cuda()
        
        h = tuple([Variable(each.data, volatile=True) for each in h])
        out, h = self.forward(inputs, h)

        p = F.softmax(out).data
        if cuda:
            p = p.cpu()
        p = p.numpy().squeeze()
        
        if top_k is not None:
            p[np.argsort(p)[:-top_k]] = 0
            p = p/p.sum()
            
        char = np.random.choice(np.arange(len(self.chars)), p=p)
            
        return self.int2char[char], h
    
    def init_weights(self):
        
        initrange = 0.1
        # Embedding weights as random uniform
        #self.embed.weight.data.uniform_(-initrange, initrange)
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)
        
    def init_hidden(self, n_seqs):
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.n_layers, n_seqs, self.n_hidden).zero_()),
                Variable(weight.new(self.n_layers, n_seqs, self.n_hidden).zero_()))


def save_model(model, filename='rnn.ckpt'):
    checkpoint = {'n_hidden': model.n_hidden,
                  'n_layers': model.n_layers,
                  'state_dict': model.state_dict(),
                  'tokens': model.chars}
    with open(filename, 'wb') as f:
        torch.save(checkpoint, f)

def load_model(filename):
    
    with open(filename, 'rb') as f:
        checkpoint = torch.load(f)

    model = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    model.load_state_dict(checkpoint['state_dict'])

    return model

def sample(model, size, prime='The', top_k=None, cuda=False):
    """ Sample characters from the model.

    """

    if cuda:
        model.cuda()
    else:
        model.cpu()

    model.eval()
    chars = [ch for ch in prime]
    h = model.init_hidden(1)
    for ch in prime:
        char, h = model.predict(ch, h, cuda=cuda, top_k=top_k)

    chars.append(char)

    for ii in range(size):
        char, h = model.predict(chars[-1], h, cuda=cuda, top_k=top_k)
        chars.append(char)

    return ''.join(chars)