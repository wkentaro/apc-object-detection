#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os.path as osp
import pickle

from chainer import cuda
from chainer import Variable
import cv2
import numpy as np

from apc_od.imaging import tile_slices_to_image


def atleast_4d(x):
    x = np.atleast_3d(x)
    if len(x.shape) == 3:
        x = np.array([x])
    return x


def tile_ae_encoded(z_data, filename):
    z_data = atleast_4d(z_data)
    for i, zi in enumerate(z_data):
        tile_img = np.array(tile_slices_to_image(zi))
        base, ext = osp.splitext(filename)
        filename_ = '{base}_{id}{ext}'.format(base=base, id=i, ext=ext)
        cv2.imwrite(filename_, tile_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('x')
    parser.add_argument('model')
    parser.add_argument('-O', '--output', default='x_encoded.jpg')
    args = parser.parse_args()

    with open(args.x) as f:
        x = pickle.load(f)
    with open(args.model) as f:
        model = pickle.load(f)

    filename = args.output

    tile_ae_encoded(model, x, filename)


if __name__ == '__main__':
    main()
