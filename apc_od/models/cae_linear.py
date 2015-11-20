#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer.functions as F
from chainer import FunctionSet
from chainer import Variable


class CAELinear(FunctionSet):
    """Convolutional Auto Encoder with Linear"""

    def __init__(self, in_units, hidden_imsize):
        super(CAELinear, self).__init__(
            conv1_1=F.Convolution2D(3, 32, 2, stride=1, pad=1),
            conv1_2=F.Convolution2D(32, 2, 2, stride=1),
            linear1_1=F.Linear(in_units, hidden_imsize**2),
            linear1_2=F.Linear(hidden_imsize**2, 10),
            linear2_1=F.Linear(10, hidden_imsize**2),
            linear2_2=F.Linear(hidden_imsize**2, in_units),
            deconv2_1=F.Deconvolution2D(2, 32, 2, stride=1),
            deconv2_2=F.Deconvolution2D(32, 3, 2, stride=1, pad=1),
        )
        self.hidden_imsize = hidden_imsize

    def forward(self, x_data, train=True):
        x = Variable(x_data, volatile=not train)
        h = self.encode(x)
        h = self.decode(h)
        return F.mean_squared_error(x, h), h

    def encode(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = self.linear1_1(h)
        h = self.linear1_2(h)
        return h

    def decode(self, x):
        n, _, = x.data.shape
        h = self.linear2_1(x)
        h = self.linear2_2(h)
        h = F.reshape(h, (n, -1, self.hidden_imsize, self.hidden_imsize))
        h = self.deconv2_1(h)
        h = self.deconv2_2(h)
        return h
