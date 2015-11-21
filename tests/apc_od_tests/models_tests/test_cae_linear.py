#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import optimizers
from chainer import Variable
import numpy as np
from skimage.data import astronaut
from skimage.transform import resize

from apc_od import im_to_blob
from apc_od.models import CAELinear


def test_cae_linear_train():
    img = astronaut()
    img = resize(img, (10, 10))
    x = np.array([im_to_blob(img)])

    model = CAELinear(in_units=200, hidden_imsize=10)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    optimizer.zero_grads()
    loss, x_hat = model.forward(x)
    loss.backward()


def test_cae_linear_encode():
    img = astronaut()
    img = resize(img, (10, 10))
    x_data = np.array([im_to_blob(img)])
    x = Variable(x_data, volatile=True)

    model = CAELinear(in_units=200, hidden_imsize=10)
    model.encode(x)


def test_cae_linear_decode():
    img = astronaut()
    img = resize(img, (10, 10))
    x_data = np.array([im_to_blob(img)])
    x = Variable(x_data, volatile=True)

    model = CAELinear(in_units=200, hidden_imsize=10)
    h = model.encode(x)
    model.decode(h)
