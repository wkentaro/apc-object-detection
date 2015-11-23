#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


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


def snack():
    here = osp.dirname(osp.abspath(__file__))
    return imread(osp.join(here, 'data/snack.jpg'))


def doll():
    here = osp.dirname(osp.abspath(__file__))
    return imread(osp.join(here, 'data/doll.jpg'))
