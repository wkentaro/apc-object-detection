#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp

from nose.tools import assert_equal


def assert_equal_path_basename(filename_a, filename_b, ignore_ext=True):
    if ignore_ext:
        filename_a = osp.splitext(filename_a)[0]
        filename_b = osp.splitext(filename_b)[0]

    assert_equal(osp.basename(filename_a), osp.basename(filename_b))
