#!/usr/bin/env python
# -*- coding: utf-8 -*-
# copied from https://github.com/mitmul/chainer-cifar10

import chainer
import chainer.functions as F
import chainer.links as L


class VGG_mini_ABN(chainer.Chain):
    """VGGnet for CIFAR-10"""

    def __init__(self):
        super(VGG_mini_ABN, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            bn1_1=L.BatchNormalization(64, decay=0.9, eps=1e-5),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),
            bn1_2=L.BatchNormalization(64, decay=0.9, eps=1e-5),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            bn2_1=L.BatchNormalization(128, decay=0.9, eps=1e-5),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            bn2_2=L.BatchNormalization(128, decay=0.9, eps=1e-5),

            fc3=L.Linear(4096, 1024),
            fc4=L.Linear(1024, 1024),
            fc5=L.Linear(1024, 3)
        )
        self.name = 'VGG_mini_ABN'

    def __call__(self, x, y):
        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, ratio=0.25)

        h = F.relu(self.bn2_1(self.conv2_1(h)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, ratio=0.25)

        h = F.dropout(F.relu(self.fc3(h)), ratio=0.5)
        h = F.dropout(F.relu(self.fc4(h)), ratio=0.5)
        h = self.fc5(h)

        return F.softmax_cross_entropy(h, y), F.accuracy(h, y), h
