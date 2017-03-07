import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers):
        super(RNNModel, self).__init__()
        if rnn_type in ['LSTM', 'RNN']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, bias=True)
        else:
            assert 0, "Only LSTM and RNN types are supported"

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        pass

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())