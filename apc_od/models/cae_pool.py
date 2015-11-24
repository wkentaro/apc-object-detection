#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L


class CAEPool(chainer.Chain):
    """Convolutional Autoencoder with Pooling"""

    def __init__(self):
        super(CAEPool, self).__init__(
            conv1=L.Convolution2D(3, 64, 2, stride=2, pad=1),
            deconv2=L.Deconvolution2D(64, 3, 2, stride=2, pad=1,
                                      outsize=(178, 267)),
        )
        self.name = 'cae_pool'
        self.y = None
        self.loss = None

    def __call__(self, x):
        z = self.encode(x)
        self.y = self.decode(z)
        self.loss = F.mean_squared_error(x, self.y)
        return self.loss, self.y

    def encode(self, x):
        h = self.conv1(x)
        z = F.max_pooling_2d(h, ksize=2, stride=2)
        return z

    def decode(self, z):
        h = F.unpooling_2d(z, ksize=2, stride=2)
        y = self.deconv2(h)
        return y
