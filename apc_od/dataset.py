#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from sklearn.datasets import load_files


here = os.path.dirname(os.path.abspath(__file__))


def get_raw(which_set):
    if which_set not in ('train', 'test'):
        raise ValueError

    data_dir = os.path.join(here, '../data/raw_{0}'.format(which_set))
    data = load_files(data_dir, load_content=False, shuffle=False)
    return data


def raw_to_mask_path(raw_path):
    raw_path = os.path.realpath(raw_path)
    raw_path = raw_path.split('/')
    raw_path[-3] = 'mask'
    mask_path = '/'.join(raw_path)
    return mask_path
