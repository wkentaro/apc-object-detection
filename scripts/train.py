#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import logging
import os
import os.path as osp
import pickle

from chainer import cuda
from chainer import optimizers
import cv2
import numpy as np
from skimage.transform import resize

from apc_od import get_raw
from apc_od.models import CAE
from draw_loss import draw_loss_curve


here = osp.dirname(osp.abspath(__file__))


def im_preprocess(im):
    shape = np.array(im.shape[:2]) // 2
    im = resize(im, shape, preserve_range=True)
    return im


def im_to_blob(im):
    """Convert image to blob.

    @param im: its shape is (height, width, channel)
    @type im: numpy.ndarray
    """
    blob = im.transpose((2, 0, 1))
    blob = blob.astype(np.float32)
    blob /= 255.
    return blob


class UnsupervisedTrain(object):

    def __init__(self, on_gpu=True):
        self.on_gpu = on_gpu
        self.timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # setup for logging
        self.log_dir = osp.join(
            here, '../logs/{}_unsupervised_train'.format(self.timestamp))
        os.mkdir(self.log_dir)
        self.log_file = osp.join(self.log_dir, 'log.txt')
        logging.basicConfig(
            format='%(asctime)s [%(levelname)s] %(message)s',
            filename=self.log_file,
            level=logging.DEBUG,
        )
        msg = 'logging at {}'.format(self.timestamp)
        logging.info(msg)
        print(msg)
        # model
        self.model = CAE()
        if self.on_gpu:
            self.model.to_gpu()
        # optimizer
        self.optimizer = optimizers.Adam(alpha=0.001)
        self.optimizer.setup(self.model)

    def batch_loop(self, dataset, train, batch_size=10):
        files = dataset.filenames
        N = len(files)
        # train loop
        sum_loss = 0
        perm = np.random.permutation(N)
        for i in range(0, N, batch_size):
            files_batch = files[perm[i:i + batch_size]]
            x_batch = np.array([im_to_blob(im_preprocess(cv2.imread(f)))
                                for f in files_batch])
            if self.on_gpu:
                x_batch = cuda.to_gpu(x_batch.astype(np.float32))
            self.optimizer.zero_grads()
            loss, x_hat = self.model.forward(x_batch, train=train)
            loss.backward()
            self.optimizer.update()
            sum_loss += float(loss.data)
        return sum_loss, x_hat

    def main_loop(self, n_epoch=10, save_interval=None):
        save_interval = save_interval or (n_epoch // 10)
        train_data = get_raw(which_set='train')
        test_data = get_raw(which_set='test')
        N_train = len(train_data.filenames)
        N_test = len(test_data.filenames)
        for epoch in xrange(0, n_epoch):
            # train
            sum_loss, x_hat = self.batch_loop(train_data, train=True)
            # logging
            msg = ('epoch:{:02d}; train mean loss={};'
                   .format(epoch, sum_loss / N_train))
            logging.info(msg)
            print(msg)
            # save
            if epoch % save_interval == 0:
                print('epoch:{:02d}; saving model and x_hat'.format(epoch))
                model_path = osp.join(
                    self.log_dir, 'CAE_{}.chainermodel.pkl'.format(epoch))
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                x_hat_path = osp.join(
                    self.log_dir, 'x_hat_{}.pkl'.format(epoch))
                with open(x_hat_path, 'wb') as f:
                    pickle.dump(x_hat.data, f)

            # test
            sum_loss, _ = self.batch_loop(test_data, train=False)
            # logging
            msg = ('epoch:{:02d}; test mean loss={};'
                   .format(epoch, sum_loss / N_test))
            logging.info(msg)
            print(msg)
        draw_loss_curve(self.log_file,
                        osp.join(self.log_dir, 'loss_curve.jpg'),
                        no_acc=True)


if __name__ == '__main__':
    app = UnsupervisedTrain()
    app.main_loop()
