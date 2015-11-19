#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import optimizers
import numpy as np
from skimage.data import astronaut

from apc_od import im_to_blob
from apc_od.models import CAE


def test_cae_train():
    img = astronaut()
    x = np.array([im_to_blob(img)])

    model = CAE()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    optimizer.zero_grads()
    loss, x_hat = model.forward(x)
    loss.backward()


def test_cae_encode():
    img = astronaut()
    x = np.array([im_to_blob(img)])

    model = CAE()
    model.encode(x)


def test_cae_decode():
    img = astronaut()
    x = np.array([im_to_blob(img)])

    model = CAE()
    model.decode(x)
