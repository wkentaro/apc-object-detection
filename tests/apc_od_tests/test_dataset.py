#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nose.tools import assert_almost_equal
from nose.tools import assert_equal

import apc_od


def test_get_raw():
    train_data = apc_od.get_raw('train')
    test_data = apc_od.get_raw('test')

    N_train = len(train_data.filenames)
    N_test = len(test_data.filenames)

    assert_almost_equal(int(N_train / 0.8), int(N_test / 0.2))


def test_raw_to_mask_path():
    raw_path = '/home/wkentaro/apc-od/data/raw_train/doll/N1_0.jpg'
    expected = '/home/wkentaro/apc-od/data/mask/doll/N1_0.jpg'
    assert_equal(expected, apc_od.raw_to_mask_path(raw_path))
