#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L


class CAE(chainer.Chain):
    """Convolutional Autoencoder"""

    def __init__(self):
        super(CAE, self).__init__(
            conv1=L.Convolution2D(3, 64, 2, stride=1, pad=1),
            deconv2=L.Deconvolution2D(64, 3, 2, stride=1, pad=1),
        )
        self.name = 'convolutional_autoencoder'
        self.y = None
        self.loss = None

    def __call__(self, x):
        h = self.encode(x)
        self.y = self.decode(h)
        self.loss = F.mean_squared_error(x, self.y)
        return self.loss, self.y

    def encode(self, x):
        return self.conv1(x)

    def decode(self, x):
        return self.deconv2(x)
