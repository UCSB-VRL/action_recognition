import torch.nn as nn
from torch.autograd import Variable
import torch
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import numpy as np


def _get_orthogonal_init_weights(weights):
    fan_out = weights.size(0)
    fan_in = weights.size(1)

    u, _, v = svd(normal(0.0, 1.0, (fan_out, fan_in)), full_matrices=False)

    if u.shape == (fan_out, fan_in):
        return torch.Tensor(u.reshape(weights.size()))
    else:
        return torch.Tensor(v.reshape(weights.size()))


def _get_xavier_init_weights(params):
    weight_shape = list(params.weight.data.size())
    fan_in = weight_shape[1]
    fan_out = weight_shape[0]
    w_bound = np.sqrt(6. / (fan_in + fan_out))
    params.weight.data.uniform_(-w_bound, w_bound)
    params.bias.data.fill_(0)
    nn.LSTM

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers, noutput, dropout, init_type):
        super(RNNModel, self).__init__()
        if rnn_type in ['LSTM', 'RNN']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, bias=True, dropout=dropout)
            if nlayers < 2 and dropout > 0.0:
                print 'INFO : With LSTM  of %d layers in LSTM, only decoder dropout will take effect'%(nlayers)
        else:
            assert 0, "Only LSTM and RNN types are supported"
        self.decoder = nn.Linear(nhid, noutput)
        self.drop = nn.Dropout(dropout)
        self.init_weights(init_type)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.noutput = noutput


    def init_weights(self, init_type='uniform'):
        assert init_type in ['uniform', 'orthogonal', 'xavier']
        if init_type == 'uniform':
            initrange = 0.1
            self.decoder.bias.data.fill_(0)
            self.decoder.weight.data.uniform_(-initrange, initrange)
        elif init_type == 'orthogonal':
            self.decoder.weight.data.copy_(_get_orthogonal_init_weights(self.decoder.weight.data))
            self.decoder.bias.data.fill_(0)
        else:
            _get_xavier_init_weights(self.decoder)




    def forward(self, input_, hidden):
        output, hidden = self.rnn(input_, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
