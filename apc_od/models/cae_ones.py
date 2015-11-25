#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import Variable


class CAEOnes(chainer.Chain):
    """Convolutional Autoencoder for Ones"""

    def __init__(self):
        super(CAEOnes, self).__init__(
            conv1=L.Convolution2D(3, 64, 2, stride=2, pad=1),
            linear1_1=L.Linear(192960, 1000),
            linear1_2=L.Linear(1000, 4),
            linear2_1=L.Linear(4, 1000),
            linear2_2=L.Linear(1000, 192960),
            deconv2=L.Deconvolution2D(64, 3, 2, stride=2, pad=1,
                                      outsize=(178, 267)),
        )
        self.name = 'cae_ones'
        self.y = None
        self.loss = None
        self.pool1_outshape = None

    def __call__(self, x):
        # encode
        z = self.encode(x)
        xp = cuda.get_array_module(z.data)
        z_t = Variable(xp.ones_like(z.data))
        loss_z = F.mean_squared_error(z, z_t)
        # decode
        self.y = self.decode(z)
        loss_y = F.mean_squared_error(x, self.y)
        self.loss = loss_z + loss_y
        return self.loss, self.y

    def encode(self, x):
        h = self.conv1(x)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        self.pool1_outshape = h.data.shape
        h = self.linear1_1(h)
        z = self.linear1_2(h)
        return z

    def decode(self, z):
        h = self.linear2_1(z)
        h = self.linear2_2(h)
        h = F.reshape(h, self.pool1_outshape)
        h = F.unpooling_2d(h, ksize=2, stride=2)
        y = self.deconv2(h)
        return y
