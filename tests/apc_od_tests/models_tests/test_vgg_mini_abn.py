#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from chainer import optimizers as O
from chainer import Variable
import numpy as np

from skimage.transform import resize

from apc_od.data import snack
from apc_od import im_to_blob
from apc_od.models import VGG_mini_ABN


class TestVGG_mini_ABN(unittest.TestCase):

    def setUp(self):
        self.model = VGG_mini_ABN()
        self.optimizer = O.Adam()
        self.optimizer.setup(self.model)

        img = snack()
        img = resize(img, (128, 128), preserve_range=True)
        x_data = np.array([im_to_blob(img)])
        y_data = np.array([0], dtype=np.int32)
        self.x = Variable(x_data)
        self.y = Variable(y_data)

    def test_train(self):
        self.optimizer.zero_grads()
        loss, accuracy, _ = self.model(self.x, self.y, train=True)
        loss.backward()
        self.optimizer.update()
