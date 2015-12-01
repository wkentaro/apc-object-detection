#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def atleast_4d(x):
    x = np.atleast_3d(x)
    if len(x.shape) == 3:
        x = np.array([x])
    return x

