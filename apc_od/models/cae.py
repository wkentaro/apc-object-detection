#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer.functions as F
from chainer import FunctionSet
from chainer import Variable


class CAE(FunctionSet):
    """Convolutional Auto Encoder"""

    def __init__(self):
        super(CAE, self).__init__(
            conv1=F.Convolution2D(3, 64, 2, stride=1, pad=1),
            pool1=F.MaxPooling2D(2, stride=1),

            unpool2=F.Unpooling2D(2, stride=1),
            deconv2=F.Deconvolution2D(64, 3, 2, stride=1, pad=1),
        )

    def forward(self, x_data, train=True):
        x = Variable(x_data, volatile=not train)

        h = self.conv1(x)
        h = self.pool1(h)

        h = self.unpool2(h)
        h = self.deconv2(h)

        return F.mean_squared_error(x, h), h
