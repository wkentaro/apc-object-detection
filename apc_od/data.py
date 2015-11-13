#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from sklearn.datasets import load_files


here = os.path.dirname(os.path.abspath(__file__))


def get_original(which_set):
    if which_set not in ('raw', 'mask'):
        raise ValueError

    data_dir = os.path.join(here, '../data/original', which_set)
    data = load_files(data_dir, load_content=False, shuffle=False)
    return data


def get_resized(which_set):
    if which_set not in ('raw', 'mask'):
        raise ValueError

    data_dir = os.path.join(here, '../data/resized', which_set)
    data = load_files(data_dir, load_content=False, shuffle=False)
    return data
