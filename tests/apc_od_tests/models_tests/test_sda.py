#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import optimizers
from chainer import Variable
import numpy as np
from skimage.data import astronaut

from apc_od import im_to_blob
from apc_od.models import StackedAutoEncoder


def test_sda_train():
    img = astronaut()
    x = np.array([im_to_blob(img)])
    x = x.reshape((len(x), -1))

    model = StackedAutoEncoder()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    optimizer.zero_grads()
    loss, x_hat = model.forward(x)
    loss.backward()


def test_sda_encode():
    img = astronaut()
    x_data = np.array([im_to_blob(img)])
    x = Variable(x_data, volatile=True)

    model = StackedAutoEncoder()
    model.encode(x)


def test_sda_decode():
    img = astronaut()
    x_data = np.array([im_to_blob(img)])
    x = Variable(x_data, volatile=True)

    model = StackedAutoEncoder()
    h = model.encode(x)
    model.decode(h)
