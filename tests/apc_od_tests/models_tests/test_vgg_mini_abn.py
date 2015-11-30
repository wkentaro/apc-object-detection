#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from chainer import optimizers as O
from chainer import Variable
from nose.tools import assert_is_not_none
import numpy as np

from skimage.transform import resize

from apc_od.data import doll
from apc_od import im_to_blob
from apc_od.models import VGG_mini_ABN


class TestVGG_mini_ABN(unittest.TestCase):

    def setUp(self):
        self.model = VGG_mini_ABN()
        self.optimizer = O.Adam()
        self.optimizer.setup(self.model)

        img = doll()
        img = resize(img, (128, 128), preserve_range=True)
        x_data = np.array([im_to_blob(img)])
        y_data = np.array([0], dtype=np.int32)
        self.x = Variable(x_data)
        self.y = Variable(y_data)

    def test_train(self):
        self.optimizer.zero_grads()
        self.model.train = True
        self.optimizer.update(self.model, self.x, self.y)
        assert_is_not_none(self.model.loss)
        assert_is_not_none(self.model.accuracy)
        assert_is_not_none(self.model.y)
