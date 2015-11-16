#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle

from chainer import cuda
import cupy
import cv2
import numpy as np

from apc_od import blob_to_im
from apc_od import tile_slices_to_image_uint8


def tile_ae_inout(x, x_hat, output_file):
    if isinstance(x, cupy.ndarray):
        x = cuda.to_cpu(x)
    if isinstance(x_hat, cupy.ndarray):
        x_hat = cuda.to_cpu(x_hat)

    X = np.vstack((x, x_hat))

    # tile images
    imgs = np.array([blob_to_im(xi) for xi in X])
    tiled_img = tile_slices_to_image_uint8(imgs)
    tiled_img = np.array(tiled_img)  # PIL image -> numpy.ndarray

    # save tiled image
    cv2.imwrite(output_file, tiled_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('x_pkl')
    parser.add_argument('x_hat_pkl')
    parser.add_argument('-O', '--output', required=True)
    args = parser.parse_args()

    # command line args
    x_pkl = args.x_pkl
    x_hat_pkl = args.x_hat_pkl
    output = args.output

    # load x
    with open(x_pkl, 'rb') as f:
        x = pickle.load(f)

    # load x_hat
    with open(x_hat_pkl, 'rb') as f:
        x_hat = pickle.load(f)

    tile_ae_inout(x, x_hat, output)


if __name__ == '__main__':
    main()
