#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os.path as osp

import numpy as np
from skimage.io import imread
from skimage.measure import regionprops
from skimage.transform import resize


def im_preprocess(im):
    shape = np.array(im.shape[:2]) // 2
    im = resize(im, shape, preserve_range=True)
    return im


def im_to_blob(im):
    """Convert image to blob.

    @param im: its shape is (height, width, channel)
    @type im: numpy.ndarray
    """
    blob = im.transpose((2, 0, 1))
    blob = blob.astype(np.float32)
    blob /= 255.
    return blob


def blob_to_im(blob):
    """Convert blob to image.

    @param blob: its shape is (channel, height, width)
    @type blob: numpy.ndarray
    """
    im = blob * 255.
    im = im.transpose((1, 2, 0))
    im = im.astype(np.uint8)
    return im


def mask_to_roi(mask):
    prop = regionprops(mask)[0]
    y0, x0 = prop.centroid
    x1 = x0 + math.cos(prop.orientation) * 0.5 * prop.major_axis_length
    y1 = y0 - math.sin(prop.orientation) * 0.5 * prop.major_axis_length
    x2 = x0 - math.sin(prop.orientation) * 0.5 * prop.minor_axis_length
    y2 = y0 - math.cos(prop.orientation) * 0.5 * prop.minor_axis_length
    return y1, x1, y2, x2


def doll():
    here = osp.dirname(osp.abspath(__file__))
    return imread(osp.join(here, 'data/doll.jpg'))


def doll_mask():
    here = osp.dirname(osp.abspath(__file__))
    return imread(osp.join(here, 'data/doll_mask.jpg'))
