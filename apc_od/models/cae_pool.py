#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer.functions as F
from chainer import FunctionSet
from chainer import Variable


class CAEPool(FunctionSet):
    """Convolutional Auto Encoder with Pooling"""

    def __init__(self):
        super(CAEPool, self).__init__(
            conv1=F.Convolution2D(3, 64, 2),
            pool1=F.MaxPooling2D(2),
            unpool2=F.Unpooling2D(2),
            deconv2=F.Deconvolution2D(64, 3, 2),
        )

    def forward(self, x_data, train=True):
        x = Variable(x_data, volatile=not train)
        h = self.encode(x)
        h = self.decode(h)
        return F.mean_squared_error(x, h), h

    def encode(self, x):
        h = self.conv1(x)
        h = self.pool1(h)
        return h

    def decode(self, x):
        h = self.unpool2(x)
        h = self.deconv2(h)
        return h
