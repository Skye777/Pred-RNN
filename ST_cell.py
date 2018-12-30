from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np

import math


class SpatioTemporal_LSTM(nn.Module):
    """docstring for SpatioTemporal_LSTM"""

    def __init__(self, input_dim, hidden_dim, batch_size, height, width, bias):
        super(SpatioTemporal_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.bias = bias
        self.kernel_size = (3, 3)
        self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2

        self.weight_1x1 = Parameter(torch.Tensor(1, 1))

        self.conv_in = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                 out_channels=self.hidden_dim,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding,
                                 bias=bias)

        self.conv_o = nn.Conv2d(in_channels=self.input_dim + 3 * self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=bias)

        self.conv_c = nn.Conv2d(in_channels=2 * self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=bias)

    def forward(self, input_tensor, state=None):
        if state is None:
            raise ValueError('nfnaiszfv vsknv')

        h_cur, c_cur, M_cur = state
        combined_conv_xh = self.conv_in(torch.cat([input_tensor, h_cur], dim=1))
        g = torch.tanh(combined_conv_xh)
        i = torch.sigmoid(combined_conv_xh)
        f = torch.sigmoid(combined_conv_xh)
        c_next = f * c_cur + i * g

        combined_conv_xM = self.conv_in(torch.cat([input_tensor, M_cur], dim=1))
        g_ = torch.tanh(combined_conv_xM)
        i_ = torch.sigmoid(combined_conv_xM)
        f_ = torch.sigmoid(combined_conv_xM)
        M_next = f_ * M_cur + i_ * g_

        o = torch.sigmoid(self.conv_o(torch.cat([input_tensor, M_next, c_next, h_cur], dim=1)))
        h_next = o * torch.tanh(self.conv_c(torch.cat([c_next, M_next], dim=1)))

        return h_next, c_next, M_next
