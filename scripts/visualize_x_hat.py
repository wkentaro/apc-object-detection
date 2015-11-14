#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle

import cv2
from chainer import cuda
import cupy
import numpy as np
from pylearn2.packaged_dependencies.theano_linear.imaging import\
    tile_slices_to_image_uint8

from apc_od import blob_to_im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('x_hat_pkl')
    parser.add_argument('-O', '--output', required=True)
    args = parser.parse_args()

    # command line args
    pkl_file = args.x_hat_pkl
    output = args.output

    # load x_hat
    with open(pkl_file, 'rb') as f:
        x_hat = pickle.load(f)
    if isinstance(x_hat, cupy.ndarray):
        x_hat = cuda.to_cpu(x_hat)

    # tile images
    imgs = np.array([blob_to_im(x) for x in x_hat])
    tiled_img = tile_slices_to_image_uint8(imgs)
    tiled_img = np.array(tiled_img)  # PIL image -> numpy.ndarray

    # save tiled image
    cv2.imwrite(output, tiled_img)


if __name__ == '__main__':
    main()
