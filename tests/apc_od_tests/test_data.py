#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nose.tools import assert_equal

import apc_od
from apc_od import testing


def test_get_original():
    raw_data = apc_od.get_original('raw')
    mask_data = apc_od.get_original('mask')

    N = len(raw_data.filenames)
    assert_equal(N, len(mask_data.filenames))

    for i in xrange(N):
        testing.assert_equal_path_basename(raw_data.filenames[i],
                                           mask_data.filenames[i])
