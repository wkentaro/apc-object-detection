#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import serializers
from chainer import Variable
import cupy
import numpy as np
from skimage.transform import resize

from apc_od import blob_to_im
from apc_od import im_to_blob
from apc_od.models.cae_ones import CAEOnes
from apc_od.models.vgg_mini_abn import VGG_mini_ABN


class CAEOnesRoiVGG(chainer.Chain):

    def __init__(self, initial_roi, cae_ones_h5, vgg_h5,
                 learning_rate=0.1, learning_n_sample=10000):
        super(CAEOnesRoiVGG, self).__init__(
            cae_ones1=CAEOnes(),
            vgg2=VGG_mini_ABN(),
        )
        self.initial_roi = initial_roi
        serializers.load_hdf5(cae_ones_h5, self.cae_ones1)
        serializers.load_hdf5(vgg_h5, self.vgg2)
        self.learning_rate = learning_rate
        self.learning_n_sample = learning_n_sample

        self.train = True
        self.y = None
        self.loss = None
        self.accuracy = None
        self.pred = None

    def __call__(self, x, t=None):
        on_gpu = isinstance(x.data, cupy.ndarray)
        self.cae_ones1.train = self.train
        self.vgg2.train = self.train

        roi_scale = self.cae_ones1.encode(x)

        if t is None:
            assert self.train is False
            if on_gpu:
                roi_scale_data = cuda.to_cpu(roi_scale.data)
            rois_data = self.initial_roi * roi_scale_data
            if on_gpu:
                rois_data = cuda.to_cpu(rois_data)
            x_data = cuda.to_cpu(x.data)
            rois_data = rois_data.astype(int)
            cropped = []
            for i in xrange(len(x_data)):
                roi = rois_data[i]
                im = blob_to_im(x_data[i])
                im = im[roi[0]:roi[2], roi[1]:roi[3]]
                im = resize(im, (128, 128), preserve_range=True)
                cropped.append(im_to_blob(im))
            cropped = np.array(cropped, dtype=np.float32)
            if on_gpu:
                cropped = cuda.to_gpu(cropped)
            cropped = Variable(cropped, volatile=not self.train)
            self.vgg2(cropped)
            self.y = self.vgg2.y
            return self.y

        # randomly change the param and estimate good parameter for the task
        min_y = None
        rands_shape = [self.learning_n_sample] + list(roi_scale.data.shape)
        rands = self.learning_rate * (2 * np.random.random(rands_shape) - 1) + 1
        rands[0] = np.ones(roi_scale.data.shape)
        for i, rand in enumerate(rands):
            if on_gpu:
                roi_scale_data = cuda.to_cpu(roi_scale.data)
            rois_data = rand * (self.initial_roi * roi_scale_data)
            x_data = cuda.to_cpu(x.data)
            skip = False
            rois_data = rois_data.astype(int)
            cropped = []
            for j in xrange(len(x_data)):
                roi = rois_data[j]
                im = blob_to_im(x_data[j])
                im = im[roi[0]:roi[2], roi[1]:roi[3]]
                if im.size == 0:
                    skip = True
                    break
                im = resize(im, (128, 128), preserve_range=True)
                cropped.append(im_to_blob(im))
            if skip:
                continue

            cropped = np.array(cropped)
            if on_gpu:
                cropped = cuda.to_gpu(cropped)
            cropped = Variable(cropped, volatile=not self.train)
            self.vgg2(cropped, t)
            h = self.vgg2.y
            loss = F.softmax_cross_entropy(h, t)
            if min_y is None:
                min_loss_data = float(loss.data)
                min_y = h
                min_loss = loss
                min_rand = rand
                min_rois = rois_data
            elif min_loss_data > float(loss.data):
                min_loss_data = float(loss.data)
                min_y = h
                min_loss = loss
                min_rand = rand
                min_rois = rois_data

        if on_gpu:
            min_rand = cuda.to_gpu(min_rand)
        rois_data = min_rand * roi_scale.data
        xp = cuda.get_array_module(rois_data)
        rois_data = rois_data.astype(xp.float32)
        rois = Variable(rois_data, volatile=not self.train)
        loss1 = F.mean_squared_error(roi_scale, rois)

        loss2 = min_loss

        self.loss = loss1 + loss2
        self.accuracy = F.accuracy(min_y, t)
        return self.loss
