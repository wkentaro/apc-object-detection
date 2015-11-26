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
            conv1_1=L.Convolution2D(3, 64, 3, stride=2, pad=1),
            conv1_2=L.Convolution2D(64, 16, 3, stride=2, pad=1),
            linear1_1=L.Linear(11616, 1000),
            linear1_2=L.Linear(1000, 4),
            linear2_1=L.Linear(4, 1000),
            linear2_2=L.Linear(1000, 11616),
            deconv2_1=L.Deconvolution2D(16, 64, 3, stride=2, pad=1),
            deconv2_2=L.Deconvolution2D(64, 3, 3, stride=2, pad=1),
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
        # conv1_1
        self.conv1_1_inshape = h.data.shape
        h = self.conv1_1(h)
        # conv1_2
        self.conv1_2_inshape = h.data.shape
        h = self.conv1_2(h)
        # pool1
        self.pool1_inshape = h.data.shape
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        self.pool1_outshape = h.data.shape
        # linears
        h = self.linear1_1(h)
        h = self.linear1_2(h)
        z = h
        return z

    def decode(self, z):
        h = z
        # linears
        h = self.linear2_1(h)
        h = self.linear2_2(h)
        # unpool1
        h = F.reshape(h, self.pool1_outshape)
        h = F.unpooling_2d(h, ksize=3, stride=2,
                           outsize=self.pool1_inshape[-2:])
        # deconv2_1
        self.deconv2_1.outsize = self.conv1_2_inshape[-2:]
        h = self.deconv2_1(h)
        # deconv2_2
        self.deconv2_2.outsize = self.conv1_1_inshape[-2:]
        h = self.deconv2_2(h)
        y = h
        return y
