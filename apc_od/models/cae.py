#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L


class CAE(chainer.Chain):
    """Convolutional Autoencoder"""

    def __init__(self):
        super(CAE, self).__init__(
            conv1=L.Convolution2D(3, 64, 2, stride=2, pad=1),
            deconv2=L.Deconvolution2D(64, 3, 2, stride=2, pad=1,
                                      outsize=(178, 267)),
        )
        self.train = True
        self.name = 'cae'
        self.z = None
        self.y = None
        self.loss = None

    def __call__(self, x):
        self.z = self.encode(x)
        self.y = self.decode(self.z)
        self.loss = F.mean_squared_error(x, self.y)
        return self.loss

    def encode(self, x):
        return self.conv1(x)

    def decode(self, x):
        return self.deconv2(x)
