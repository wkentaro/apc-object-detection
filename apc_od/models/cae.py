#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable, FunctionSet
import chainer.functions as F


class CAE(FunctionSet):
    """Convolutional Auto Encoder"""

    def __init__(self):
        super(CAE, self).__init__(
            conv1=F.Convolution2D(3, 64, 3, stride=1, pad=1),
            pool1=F.MaxPooling2D(2, stride=1),

            conv2=F.Convolution2D(64, 128, 3, stride=1, pad=1),
            pool2=F.MaxPooling2D(2, stride=1),

            unpool3=F.Unpooling2D(2, stride=1),
            deconv3=F.Deconvolution2D(128, 64, 3, stride=1, pad=1),

            unpool4=F.Unpooling2D(2, stride=1),
            deconv4=F.Deconvolution2D(64, 3, 3, stride=1, pad=1),
        )

    def forward(self, x_data, train=True):
        x = Variable(x_data, volatile=not train)

        h = self.conv1(x)
        h = self.pool1(h)

        h = self.conv2(h)
        h = self.pool2(h)

        h = self.unpool3(h)
        h = self.deconv3(h)

        h = self.unpool4(h)
        h = self.deconv4(h)

        return F.mean_squared_error(x, h), h
