#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import optimizers
from chainer import Variable
import numpy as np
from skimage.data import astronaut

from apc_od import im_to_blob
from apc_od.models import CAE


def test_cae_train():
    img = astronaut()
    x = np.array([im_to_blob(img)])
    x = Variable(x)

    model = CAE()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    optimizer.zero_grads()
    loss, x_hat = model(x)
    loss.backward()
    optimizer.update()


def test_cae_encode():
    img = astronaut()
    x = np.array([im_to_blob(img)])
    x = Variable(x)

    model = CAE()
    model.encode(x)


def test_cae_decode():
    img = astronaut()
    x = np.array([im_to_blob(img)])
    x = Variable(x)

    model = CAE()
    h = model.encode(x)
    model.decode(h)
