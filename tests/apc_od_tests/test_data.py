#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nose.tools import assert_almost_equal

import apc_od


def test_get_raw():
    train_data = apc_od.get_raw('train')
    test_data = apc_od.get_raw('test')

    N_train = len(train_data.filenames)
    N_test = len(test_data.filenames)

    assert_almost_equal(int(N_train / 0.8), int(N_test / 0.2))
