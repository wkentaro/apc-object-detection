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
TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%s')
LOG_DIR = osp.join(here, '../logs/{}_unsupervised_train'.format(TIMESTAMP))
os.mkdir(LOG_DIR)

log_file = osp.join(LOG_DIR, 'log.txt')
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename=log_file,
    level=logging.DEBUG,
)
msg = 'logging at {}'.format(TIMESTAMP)
logging.info(msg)
print(msg)

ON_GPU = True


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


def train(dataset, model):
    files = dataset.filenames

    # fixed parameters
    N = len(files)
    batch_size = 10
    adam_alpha = 0.001

    # optimizer
    optimizer = optimizers.Adam(alpha=adam_alpha)
    optimizer.setup(model)

    # train loop
    sum_loss = 0
    perm = np.random.permutation(N)
    for i in range(0, N, batch_size):
        files_batch = files[perm[i:i + batch_size]]
        x_batch = np.array([im_to_blob(im_preprocess(cv2.imread(f)))
                            for f in files_batch])
        if ON_GPU:
            x_batch = cuda.to_gpu(x_batch.astype(np.float32))

        optimizer.zero_grads()
        loss, x_hat = model.forward(x_batch, train=True)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data)

    return sum_loss, model, x_hat


def main():
    # fixed parameters
    n_epoch = 10
    save_interval = n_epoch // 10

    # model
    model = CAE()
    if ON_GPU:
        model.to_gpu()

    dataset = get_raw(which_set='train')
    N = len(dataset.filenames)

    for epoch in xrange(0, n_epoch):
        sum_loss, model, x_hat = train(dataset, model)

        # logging
        msg = 'epoch:{:02d}\ttrain mean loss={}'.format(epoch, sum_loss / N)
        logging.info(msg)
        print(msg)

        if epoch % save_interval == 0:
            print('epoch:{:02d}\tsaving model and x_hat'.format(epoch))
            model_path = osp.join(
                LOG_DIR, 'CAE_{}.chainermodel.pkl'.format(epoch))
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            x_hat_path = osp.join(LOG_DIR, 'x_hat_{}.pkl'.format(epoch))
            with open(x_hat_path, 'wb') as f:
                pickle.dump(x_hat.data, f)

        draw_loss_curve(
            log_file, osp.join(LOG_DIR, 'loss_curve.jpg'), no_acc=True)


if __name__ == '__main__':
    main()
