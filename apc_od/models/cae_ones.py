#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import Variable


class CAEOnes(chainer.Chain):
    """Convolutional Autoencoder for Ones"""

    def __init__(self, linear_size=11616, n_param=4):
        super(CAEOnes, self).__init__(
            conv1_1=L.Convolution2D(3, 8, 3, stride=2, pad=1),
            conv1_2=L.Convolution2D(8, 16, 3, stride=2, pad=1),
            linear1_1=L.Linear(linear_size, 4096),
            linear1_2=L.Linear(4096, n_param),
            linear2_1=L.Linear(n_param, 4096),
            linear2_2=L.Linear(4096, linear_size),
            deconv2_1=L.Deconvolution2D(16, 8, 3, stride=2, pad=1),
            deconv2_2=L.Deconvolution2D(8, 3, 3, stride=2, pad=1),
        )
        self.train = True
        self.z = None
        self.y = None
        self.loss = None
        self.pool1_outshape = None

    def __call__(self, x):
        # encode
        self.z = self.encode(x)
        xp = cuda.get_array_module(self.z.data)
        volatile = 'off' if self.train else 'on'
        z_t = Variable(xp.ones_like(self.z.data), volatile=volatile)
        loss_z = F.mean_squared_error(self.z, z_t)
        # decode
        self.y = self.decode(self.z)
        loss_y = F.mean_squared_error(x, self.y)
        self.loss = loss_z + loss_y
        return self.loss

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
        self.z = h
        return self.z

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
        self.y = h
        return self.y
