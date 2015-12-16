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


class Pipeline(chainer.Chain):
    pass


class CAEOnesRoiVGG(Pipeline):

    def __init__(
            self,
            initial_roi,
            cae_ones_h5,
            vgg_h5,
            learning_rate=0.1,
            learning_n_sample=100
            ):
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
        self.accuracy = None
        self.pred = None

    def x0_to_x1(self, x0, roi_scale):
        on_gpu = isinstance(x0.data, cupy.ndarray)
        roi_scale_data = cuda.to_cpu(roi_scale.data) \
            if on_gpu else roi_scale.data
        rois_data = (self.initial_roi * roi_scale_data).astype(int)
        x0_data = cuda.to_cpu(x0.data) if on_gpu else x0.data
        cropped = []
        for i in xrange(len(x0_data)):
            roi = rois_data[i]
            im = blob_to_im(x0_data[i])
            im = im[roi[0]:roi[2], roi[1]:roi[3]]
            if im.size() == 0:
                break
            im = resize(im, (128, 128), preserve_range=True)
            cropped.append(im_to_blob(im))
        else:
            cropped_data = np.array(cropped, dtype=np.float32)
            if on_gpu:
                cropped_data = cuda.to_gpu(cropped_data)
            x1 = Variable(cropped_data, volatile=not self.train)
            return x1

    def random_sample(self, x, t):
        """Randomly changes the parameters of each link.

        returns better parameters to regress for the task to get ``t``
        """
        on_gpu = isinstance(x.data, cupy.ndarray)

        roi_scale = self.cae_ones1.encode(x)

        min_y = None
        rands_shape = [self.learning_n_sample] + list(roi_scale.data.shape)
        rands = self.learning_rate * \
            (2 * np.random.random(rands_shape) - 1) + 1
        rands[0] = np.ones(roi_scale.data.shape)
        roi_scale_data = cuda.to_cpu(roi_scale.data) \
            if on_gpu else roi_scale.data
        for i, rand in enumerate(rands):
            roi_scale_data_with_rand = rand * roi_scale_data
            roi_scale = Variable(roi_scale_data_with_rand,
                                 volatile=not self.train)
            x1 = self.x0_to_x1(x0=x, roi_scale=roi_scale)
            if x1 is None:
                continue
            self.vgg2(x1, t)
            h = self.vgg2.y
            loss = F.softmax_cross_entropy(h, t)
            if min_y is None:
                min_loss_data = float(loss.data)
                min_y = h
                min_rand = rand
            elif min_loss_data > float(loss.data):
                min_loss_data = float(loss.data)
                min_y = h
                min_rand = rand

        # DEBUG
        # from skimage.io import imsave
        # timestamp = str(time.time())
        # os.mkdir(timestamp)
        # for i, (xi, roi) in enumerate(zip(x_data, min_rois)):
        #     im = blob_to_im(xi)
        #     im = im[roi[0]:roi[2], roi[1]:roi[3]]
        #     imsave('{}/{}.jpg'.format(timestamp, i), im)

        # convert from xp.ndarray to chainer.Variable
        roi_scale_data_with_rand = min_rand * roi_scale.data
        roi_scale_data_with_rand = roi_scale_data_with_rand.astype(np.float32)
        if on_gpu:
            roi_scale_data_with_rand = cuda.to_gpu(roi_scale_data_with_rand)
        roi_scale = Variable(roi_scale_data_with_rand, volatile=not self.train)

        t_0 = roi_scale
        t_n = self.x0_to_x1(x0=t, roi_scale=roi_scale)

        X_est = [t_0, t_n]
        return X_est

    def __call__(self, x, t=None):
        self.cae_ones1.train = self.train
        self.vgg2.train = self.train

        roi_scale = self.cae_ones1.encode(x)

        # just use as regression
        if t is None:
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # testing fase
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            assert self.train is False
            x1 = self.x0_to_x1(x0=x, roi_scale=roi_scale)
            self.vgg2(x1)
            self.y = self.vgg2.y
            return self.y

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # training fase
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # estimate better parameters
        X_est = self.random_sample(x, t)
        # get roi parameter
        z = self.cae_ones1.encode(x)
        # optimize roi parameter to be better
        loss1 = F.mean_squared_error(z, X_est[0])
        # optimize regression parameter to be better
        loss2 = self.vgg2(X_est[1], t)

        y = self.vgg2.y
        self.accuracy = F.accuracy(y, t)
        return loss1, loss2
