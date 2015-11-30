#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import chainer.optimizers as O
from chainer import Variable
from nose.tools import assert_is_not_none
import numpy as np

from apc_od.data import doll
from apc_od import im_preprocess
from apc_od import im_to_blob
from apc_od.models import CAEOnes


class TestCAEOnes(unittest.TestCase):

    def setUp(self):
        self.model = CAEOnes()
        self.optimizer = O.Adam()
        self.optimizer.setup(self.model)

        img = doll()
        x_data = np.array([im_to_blob(im_preprocess(img))])
        self.x = Variable(x_data)

    def test_train(self):
        self.optimizer.zero_grads()
        self.model.train = True
        self.optimizer.update(self.model, self.x)
        assert_is_not_none(self.model.z)
        assert_is_not_none(self.model.y)
        assert_is_not_none(self.model.loss)

    def test_encode(self):
        self.model.train = False
        self.model.encode(self.x)

    def test_decode(self):
        self.model.train = False
        h = self.model.encode(self.x)
        self.model.decode(h)
