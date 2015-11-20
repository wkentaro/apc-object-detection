#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer.functions as F
from chainer import FunctionSet
from chainer import Variable


class StackedAutoEncoder(FunctionSet):
    """Stacked Auto Encoder"""

    def __init__(self):
        super(StackedAutoEncoder, self).__init__(
            linear1=F.Linear(786432, 10),
            linear2=F.Linear(10, 786432),
        )

    def forward(self, x_data, train=True):
        x = Variable(x_data, volatile=not train)
        h = self.encode(x)
        h = self.decode(h)
        return F.mean_squared_error(x, h), h

    def encode(self, x):
        return F.relu(self.linear1(x))

    def decode(self, x):
        return F.relu(self.linear2(x))
