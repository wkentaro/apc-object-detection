#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os.path as osp
import pickle

from chainer import Variable
from chainer import cuda
import cv2
import numpy as np

from apc_od.imaging import tile_slices_to_image


def tile_ae_encoded(model, x, filename):
    x = cuda.to_gpu(x)
    x = Variable(x, volatile=True)
    h = model.conv1(x)
    for i, hi in enumerate(cuda.to_cpu(h.data)):
        tile_img = np.array(tile_slices_to_image(hi))
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
