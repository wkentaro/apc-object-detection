#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from chainer import serializers
from chainer import Variable
import numpy as np
from skimage.io import imread
from skimage.transform import resize

from apc_od import im_to_blob
from apc_od import mask_to_roi
from apc_od.models import VGG_mini_ABN
from apc_od import raw_to_mask_path


OBJECT_CLASSES = ('doll', 'snack', 'doll')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_path')
    parser.add_argument('model_path')
    args = parser.parse_args()

    raw_path = args.raw_path
    model_path = args.model_path

    mask_path = raw_to_mask_path(raw_path)
    raw = imread(raw_path)
    mask = imread(mask_path)
    y_min, x_min, y_max, x_max = mask_to_roi(mask)

    im = raw[y_min:y_max, x_min:x_max]

    model = VGG_mini_ABN()
    serializers.load_hdf5(model_path, model)

    im = resize(im, (128, 128), preserve_range=True)
    x_data = np.array([im_to_blob(im)], dtype=np.float32)
    x = Variable(x_data, volatile=True)
    model.train = False
    y = model(x)
    y_data = y.data
    print(OBJECT_CLASSES[np.argmax(y_data[0])])


if __name__ == '__main__':
    main()
