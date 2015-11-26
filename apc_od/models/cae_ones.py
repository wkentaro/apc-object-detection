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
            conv1_1=L.Convolution2D(3, 64, 2, stride=2, pad=1),
            conv1_2=L.Convolution2D(64, 3, 2, stride=2, pad=1),
            linear1_1=L.Linear(2346, 1000),
            linear1_2=L.Linear(1000, 4),
            linear2_1=L.Linear(4, 1000),
            linear2_2=L.Linear(1000, 2346),
            deconv2_1=L.Deconvolution2D(3, 64, 2, stride=2, pad=1),
            deconv2_2=L.Deconvolution2D(64, 3, 2, stride=2, pad=1,
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
        z_t = Variable(xp.ones_like(z.data), volatile=z.volatile)
        loss_z = F.mean_squared_error(z, z_t)
        # decode
        self.y = self.decode(z)
        loss_y = F.mean_squared_error(x, self.y)
        self.loss = loss_z + loss_y
        return self.loss, self.y

    def encode(self, x):
        h = x
        h = self.conv1_1(h)
        h = self.conv1_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        self.pool1_outshape = h.data.shape
        h = self.linear1_1(h)
        h = self.linear1_2(h)
        z = h
        return z

    def decode(self, z):
        h = z
        h = self.linear2_1(h)
        h = self.linear2_2(h)
        h = F.reshape(h, self.pool1_outshape)
        h = F.unpooling_2d(h, ksize=2, stride=2)
        h = self.deconv2_1(h)
        h = self.deconv2_2(h)
        y = h
        return y
