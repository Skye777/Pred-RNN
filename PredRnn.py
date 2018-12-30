from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from ST_LSTM import *
from ST_cell import *


class Encoder(nn.Module):
    """docstring for Encoder"""

    def __init__(self, n_layers, hidden_sizes, input_sizes, batch_size, channels, height, width, bias=True, use_cuda=False):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_sizes = hidden_sizes
        self.input_sizes = input_sizes
        self.channels = channels
        self.bias = bias
        self.use_cuda = use_cuda

        self.M = dict()
        self.C = dict()
        self.H = dict()

        self.h = dict()
        self.c = dict()
        self.m = {0: Parameter(torch.Tensor())}

        for i in range(n_layers):
            # self.h[i] = Parameter(torch.Tensor([batch_size, hidden_sizes[i], height, width]))
            # self.c[i] = Parameter(torch.Tensor([batch_size, hidden_sizes[i], height, width]))
            self.h[i] = Parameter(torch.zeros(batch_size, hidden_sizes[i], height, width))
            self.c[i] = Parameter(torch.zeros(batch_size, hidden_sizes[i], height, width))
            self.m[i] = Parameter(torch.zeros(batch_size, hidden_sizes[i], height, width))

        # self.h[0] = Parameter(torch.Tensor(shape))
        # self.h[1] = Parameter(torch.Tensor(shape))
        # self.h[2] = Parameter(torch.Tensor(shape))
        # self.h[3] = Parameter(torch.Tensor(shape))
        #
        # self.c[0] = Parameter(torch.Tensor(shape))
        # self.c[1] = Parameter(torch.Tensor(shape))
        # self.c[2] = Parameter(torch.Tensor(shape))
        # self.c[3] = Parameter(torch.Tensor(shape))

        self.cells = nn.ModuleList([])
        for i in range(self.n_layers):
            cur_input_dim = self.channels if i==0 else self.hidden_sizes[i-1]
            cell = SpatioTemporal_LSTM(input_dim=cur_input_dim, hidden_dim=hidden_sizes[i], batch_size=batch_size,
                                       height=height, width=width, bias=bias)
            self.cells.append(cell)

        # self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, first_timestep=False):
        for j, cell in enumerate(self.cells):
            if first_timestep:
                if j == 0:
                    # first time step and 1st layer
                    self.H[j], self.C[j], self.M[j] = cell(input_, (self.h[j], self.c[j], self.m[j]))
                    continue
                else:
                    # first time step and not 1st layer
                    self.H[j], self.C[j], self.M[j] = cell(self.H[j - 1], (self.h[j], self.c[j], self.M[j - 1]))
                continue

            if j == 0:
                # 1st layer and not 1st time step
                self.H[j], self.C[j], self.M[j] = cell(input_, (self.H[j], self.C[j], self.M[self.n_layers - 1]))
                continue

            # neither 1st
            self.H[j], self.C[j], self.M[j] = cell(self.H[j - 1], (self.H[j], self.C[j], self.M[j - 1]))

        # return self.H, self.C, self.M
        return self.H, self.C

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_sizes[0]))  #################SHAPE
        if self.use_cuda:
            return result.cuda()
        else:
            return result


class Decoder(nn.Module):
    """
    docstring for Decoder

    Using M in zigzag fashion as suggested in Spatiotemporal LSTM

    """

    def __init__(self, n_layers, hidden_sizes, input_sizes, batch_size, height, width, bias=True, use_cuda=False):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_sizes = hidden_sizes
        self.input_sizes = input_sizes
        self.use_cuda = use_cuda

        self.cells = nn.ModuleList([])
        for i in range(self.n_layers):
            cell = SpatioTemporal_LSTM(self.hidden_sizes[i], self.input_sizes, batch_size, height, width, bias)
            self.cells.append(cell)

    def forward(self, input_, C, H, M):
        for j, cell in enumerate(self.cells):
            if j == 0:
                H[j], C[j], M[j] = cell(input_, (H[j], C[j], M[self.n_layers - 1]))

            if j == self.n_layers - 1:
                H[j], C[j], M[j] = cell(H[j - 1], (H[j], C[j], M[j - 1]))
                output = H[j]
        return output

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result
