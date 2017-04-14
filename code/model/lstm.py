import torch.nn as nn
from torch.autograd import Variable
import torch
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import numpy as np

def _get_orthogonal_init_weights(size):
    u, _, v = svd(normal(0.0, 1.0, size), full_matrices=False)

    if u.shape == size:
        return torch.Tensor(u.reshape(size))
    else:
        return torch.Tensor(v.reshape(size))

class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ninp, nhid, nlayers, noutput, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(nhid, nhid, nlayers, bias=True, dropout=dropout)
        if nlayers < 2 and dropout > 0.0:
            print 'INFO : With LSTM  of %d layers in LSTM, only decoder dropout will take effect'%(nlayers)

        self.encoder = nn.Linear(ninp, nhid)
        self.decoder = nn.Linear(nhid, noutput)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.states = None
        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers
        self.noutput = noutput

    def init_weights(self):
        for weight in self.lstm.parameters():
            size = weight.size()
            if len(size) > 1:
                h, w = size
                size = (h // 4, w)
                mat = torch.cat([_get_orthogonal_init_weights(size)
                    for _ in range(4)])
                weight.data.copy_(mat)
            else:
                weight.data.fill_(0)
        self.encoder.weight.data *= sqrt(2) # due to relu


    def forward(self, x):
        n, b, d = x.size()
        h, c = self.init_states(b)
        x = self.encoder(x.view(-1, d)).view(n, b, -1)
        x = self.relu(x)
        x = self.drop1(x)
        out, (h, c) = self.lstm(x, (h, c))
        out = out.mean(0).view(b, -1)
        x = self.drop2(out)
        x = self.decoder(x)
        return x

    def init_states(self, b):
        s = self.nhid
        if self.states is None:
            w = next(self.parameters()).data
            self.states = (Variable(w.new(self.nlayers, b, s).zero_()),
                    Variable(w.new(self.nlayers, b, s).zero_()))
        elif self.states[0].size(1) != b:
            for h in self.states:
                h.data.resize_(self.nlayers, b, s).zero_()
        else:
            for h in self.states:
                h.data.zero_()
        return self.states
