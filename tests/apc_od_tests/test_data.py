#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nose.tools import assert_tuple_equal

import apc_od
from apc_od.data import doll_mask


def test_mask_to_roi():
    mask = doll_mask()
    roi = apc_od.mask_to_roi(mask)
    assert_tuple_equal(roi, (208, 216, 264, 392))
