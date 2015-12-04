#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import unittest

import chainer.optimizers as O
from chainer import Variable
from nose.tools import assert_is_not_none
import numpy as np

from apc_od.data import doll
from apc_od import im_preprocess
from apc_od import im_to_blob
from apc_od.models import CAEOnesRoiVGG


here = os.path.dirname(os.path.abspath(__file__))


class TestCAEOnesRoiVGG(unittest.TestCase):

    def setUp(self):
        cae_ones_h5 = os.path.join(here, 'data/cae_ones.h5')
        vgg_h5 = os.path.join(here, 'data/vgg.h5')
        self.model = CAEOnesRoiVGG(
            initial_roi=[100, 100, 300, 300],
            cae_ones_h5=cae_ones_h5, vgg_h5=vgg_h5)
        self.optimizer = O.Adam()
        self.optimizer.setup(self.model)

        img = doll()
        x_data = np.array([im_to_blob(im_preprocess(img))])
        self.x = Variable(x_data)
        t_data = np.array([0], dtype=np.int32)
        self.t = Variable(t_data)

    def test_train(self):
        self.optimizer.zero_grads()
        self.model.train = True
        self.optimizer.update(self.model, self.x, self.t)
        assert_is_not_none(self.model.loss)

    def test_test(self):
        self.model.train = False
        y = self.model(self.x)
