#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
import logging
import os
import os.path as osp
import pickle
import sys

from chainer import cuda
from chainer import optimizers as O
from chainer import serializers
from chainer import Variable
import cv2
import numpy as np

from apc_od import get_raw
from apc_od import im_preprocess
from apc_od import im_to_blob
from apc_od.models import CAE
from apc_od.models import CAEOnes
from apc_od.models import CAEPool
from draw_loss import draw_loss_curve
from tile_ae_encoded import tile_ae_encoded
from tile_ae_inout import tile_ae_inout


here = osp.dirname(osp.abspath(__file__))


class UnsupervisedTrain(object):

    def __init__(self, model, log_dir, log_file, on_gpu):
        self.log_dir = log_dir
        self.log_file = log_file
        self.on_gpu = on_gpu
        self.model = model
        # setup model on gpu
        if self.on_gpu:
            self.model.to_gpu()
        # optimizer
        self.optimizer = O.Adam(alpha=0.001)
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
            x = Variable(x_batch, volatile=not train)
            self.optimizer.zero_grads()
            loss, x_hat = self.model(x)
            loss.backward()
            self.optimizer.update()
            sum_loss += float(loss.data)
        x_hat_data = x_hat.data
        if self.on_gpu:
            x_batch = cuda.to_cpu(x_batch)
            x_hat_data = cuda.to_cpu(x_hat_data)
        return sum_loss, x_batch, x_hat_data

    def main_loop(self, n_epoch=10, save_interval=None, save_encoded=True):
        save_interval = save_interval or (n_epoch // 10)
        train_data = get_raw(which_set='train')
        test_data = get_raw(which_set='test')
        N_train = len(train_data.filenames)
        N_test = len(test_data.filenames)
        for epoch in xrange(0, n_epoch):
            # train
            sum_loss, _, _ = self.batch_loop(train_data, train=True)
            # logging
            msg = ('epoch:{:02d}; train mean loss={};'
                   .format(epoch, sum_loss / N_train))
            logging.info(msg)
            print(msg)
            # test
            sum_loss, x, x_hat = self.batch_loop(test_data, train=False)
            # logging
            msg = ('epoch:{:02d}; test mean loss={};'
                   .format(epoch, sum_loss / N_test))
            logging.info(msg)
            print(msg)
            # save model and input/encoded/decoded
            if epoch % save_interval == (save_interval - 1):
                print('epoch:{:02d}; saving'.format(epoch))
                # save model
                model_path = osp.join(
                    self.log_dir,
                    '{name}_model_{epoch}.h5'.format(name=self.model.name,
                                                     epoch=epoch))
                serializers.save_hdf5(model_path, self.model)
                # save x_data
                x_path = osp.join(self.log_dir, 'x_{}.pkl'.format(epoch))
                with open(x_path, 'wb') as f:
                    pickle.dump(x, f)  # save x
                x_hat_path = osp.join(
                    self.log_dir, 'x_hat_{}.pkl'.format(epoch))
                with open(x_hat_path, 'wb') as f:
                    pickle.dump(x_hat, f)  # save x_hat
                tile_ae_inout(x, x_hat,
                              osp.join(self.log_dir, 'X_{}.jpg'.format(epoch)))
                z = self.model.encode(Variable(cuda.to_gpu(x), volatile=True))
                if save_encoded:
                    tile_ae_encoded(
                        cuda.to_cpu(z.data),
                        osp.join(self.log_dir,
                                 'x_encoded_{}.jpg'.format(epoch)))

        draw_loss_curve(self.log_file,
                        osp.join(self.log_dir, 'loss_curve.jpg'),
                        no_acc=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50,
                        help='number of recursion (default: 50)')
    parser.add_argument('--model', type=str, default='CAE',
                        help='name of model (default: CAE)')
    parser.add_argument('--no-logging', action='store_true',
                        help='logging to tmp dir')
    args = parser.parse_args()

    save_encoded = True
    n_epoch = args.epoch
    if args.model == 'CAE':
        model = CAE()
    elif args.model == 'CAEPool':
        model = CAEPool()
    elif args.model == 'CAEOnes':
        model = CAEOnes()
        save_encoded = False
    else:
        sys.stderr.write('Unsupported model: {}'.format(args.model))
        sys.exit(1)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # setup for logging
    if args.no_logging:
        import tempfile
        log_dir = tempfile.mkdtemp()
    else:
        log_dir = osp.join(here, '../logs/{}_{}'.format(timestamp, args.model))
        log_dir = osp.realpath(osp.abspath(log_dir))
        os.mkdir(log_dir)
    log_file = osp.join(log_dir, 'log.txt')
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=log_file,
        level=logging.DEBUG,
    )
    logging.info('args: {};'.format(args))
    msg = 'logging in {};'.format(log_dir)
    logging.info(msg)
    print(msg)

    app = UnsupervisedTrain(
        model=model,
        log_dir=log_dir,
        log_file=log_file,
        on_gpu=True
    )
    app.main_loop(n_epoch=n_epoch, save_encoded=save_encoded)


if __name__ == '__main__':
    main()
