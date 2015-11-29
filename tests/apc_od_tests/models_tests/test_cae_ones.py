#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import chainer.optimizers as O
from chainer import Variable
import numpy as np

from apc_od.data import snack
from apc_od import im_preprocess
from apc_od import im_to_blob
from apc_od.models import CAEOnes


class TestCAEOnes(unittest.TestCase):

    def setUp(self):
        self.model = CAEOnes()
        self.optimizer = O.Adam()
        self.optimizer.setup(self.model)

        img = snack()
        x_data = np.array([im_to_blob(im_preprocess(img))])
        self.x = Variable(x_data)

    def test_train(self):
        self.optimizer.zero_grads()
        loss, x_hat = self.model(self.x, train=True)
        loss.backward()

    def test_encode(self):
        self.model.encode(self.x)

    def test_decode(self):
        h = self.model.encode(self.x)
        self.model.decode(h)
