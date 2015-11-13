#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize

from sklearn.datasets import load_files


here = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(here, 'resized')):
    print('Please copy original/ to resized/.')
    quit()


raw_files = load_files('resized/raw', shuffle=False, load_content=False)
mask_files = load_files('resized/mask', shuffle=False, load_content=False)
for rf, mf in zip(raw_files.filenames, mask_files.filenames):
    raw_img = imread(rf)
    raw_img = resize(raw_img, (712, 1068))
    imsave(rf, raw_img)
    mask_img = imread(mf)
    mask_img = resize(mask_img, (712, 1068))
    imsave(mf, mask_img)