#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import collections
import datetime
import logging
import os
import os.path as osp
import pickle
import sys

import matplotlib
matplotlib.use('Agg')

from chainer import cuda
from chainer import optimizers as O
from chainer import serializers
from chainer import Variable
import numpy as np
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize

from apc_od import draw_loss_curve
from apc_od import get_raw
from apc_od import im_preprocess
from apc_od import im_to_blob
from apc_od import mask_to_roi
from apc_od import raw_to_mask_path
from apc_od import roi_preprocess
from apc_od import tile_ae_encoded
from apc_od import tile_ae_inout


here = osp.dirname(osp.abspath(__file__))
timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
save_dir = 'test_result_' + timestamp
os.mkdir(save_dir)


def save_roi_applied(imgs, rois, name):
    for i, (img, roi) in enumerate(zip(imgs, rois)):
        roi_img = img[roi[0]:roi[2], roi[1]:roi[3]]
        if roi_img.size == 0:
            continue
        roi_img = roi_img.astype(np.uint8)
        imsave('{dir}/{i_img}_{name}.jpg'.format(dir=save_dir, i_img=i, name=name), roi_img)


def test_cae_ones_roi_vgg():
    from apc_od.pipeline import CAEOnesRoiVGG
    initial_roi = np.array([100, 130, 300, 400])
    initial_roi = roi_preprocess(initial_roi)
    model = CAEOnesRoiVGG(initial_roi)
    model.train = False
    model.cae_ones1.train = False
    serializers.load_hdf5(os.path.join(here, 'cae_ones_roi_vgg_model.h5'), model)
    dataset = get_raw('test')
    x1_data = []
    raw_imgs = []
    mask_imgs = []
    for raw_path in dataset.filenames:
        raw = im_preprocess(imread(raw_path))
        mask_path = raw_to_mask_path(raw_path)
        mask = im_preprocess(imread(mask_path))
        roi = mask_to_roi(mask)
        raw_imgs.append(raw)
        x1_data.append(im_to_blob(raw))
        mask_imgs.append(mask)
    x1_data = np.array(x1_data, dtype=np.float32)
    x1 = Variable(x1_data, volatile='on')
    roi_scale = model.cae_ones1.encode(x1)
    roi_scale = roi_scale.data

    rois_before = np.ones_like(roi_scale) * initial_roi
    save_roi_applied(raw_imgs, rois_before, name='rois_before')
    rois_after = roi_scale * initial_roi
    save_roi_applied(raw_imgs, rois_after, name='rois_after')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='name of model')
    args = parser.parse_args()

    if args.model == 'CAEOnesRoiVGG':
        test_cae_ones_roi_vgg()


if __name__ == '__main__':
    main()
