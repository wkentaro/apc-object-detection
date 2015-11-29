#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nose.tools import assert_equal

import apc_od


def test_raw_to_mask_path():
    raw_path = '/home/wkentaro/apc-od/data/raw_train/doll/N1_0.jpg'
    expected = '/home/wkentaro/apc-od/data/mask/doll/N1_0.jpg'
    assert_equal(expected, apc_od.raw_to_mask_path(raw_path))
