#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import cv2
import numpy as np

from apc_od.utils import atleast_4d
from apc_od.data import blob_to_im
from apc_od.imaging import tile_slices_to_image_uint8
from apc_od.imaging import tile_slices_to_image


def tile_ae_inout(x, x_hat, output_file):
    X = np.vstack((x, x_hat))

    # tile images
    imgs = np.array([blob_to_im(xi) for xi in X])
    tiled_img = tile_slices_to_image_uint8(imgs)
    tiled_img = np.array(tiled_img)  # PIL image -> numpy.ndarray

    # save tiled image
    cv2.imwrite(output_file, tiled_img)


def tile_ae_encoded(z_data, filename):
    z_data = atleast_4d(z_data)
    for i, zi in enumerate(z_data):
        tile_img = np.array(tile_slices_to_image(zi))
        base, ext = os.path.splitext(filename)
        filename_ = '{base}_{id}{ext}'.format(base=base, id=i, ext=ext)
        cv2.imwrite(filename_, tile_img)