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
import glob

import matplotlib
matplotlib.use('Agg')

from chainer import cuda
from chainer import optimizers as O
from chainer import serializers
from chainer import Variable
import numpy as np
from skimage.io import imread
from skimage.transform import resize

from apc_od import draw_loss_curve
from apc_od import get_inbin_depth
from apc_od import im_preprocess
from apc_od import im_to_blob
from apc_od import mask_to_roi
from apc_od import raw_to_mask_path
from apc_od import roi_preprocess
from apc_od import tile_ae_encoded
from apc_od import tile_ae_inout


def get_inhand_mask(which_set):
    return glob.glob('../data/inhand_mask_{}/*.jpg'.format(which_set))


here = osp.dirname(osp.abspath(__file__))


class Trainer(object):

    def __init__(
            self,
            optimizers,
            model,
            model_name,
            is_supervised,
            crop_roi,
            batch_size,
            log_dir,
            log_file,
            on_gpu,
            ):
        self.optimizers = optimizers
        self.model = model
        self.model_name = model_name
        self.is_supervised = is_supervised
        self.crop_roi = crop_roi
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.log_file = log_file
        self.on_gpu = on_gpu

    def batch_loop(self, x_data, t_data, train):
        N = len(x_data)
        # train loop
        sum_loss = collections.defaultdict(float)
        sum_accuracy = 0 if self.is_supervised else None
        perm = np.random.permutation(N)
        for i in range(0, N, self.batch_size):
            print('train.Trainer.batch_loop: i: {}'.format(i))
            x_batch = x_data[perm[i:i + self.batch_size]]
            t_batch = t_data[perm[i:i + self.batch_size]]
            if self.on_gpu:
                x_batch = cuda.to_gpu(x_batch)
                t_batch = cuda.to_gpu(t_batch)
            volatile = 'off' if train else 'on'
            x = Variable(x_batch, volatile=volatile)
            t = Variable(t_batch, volatile=volatile)
            for j in xrange(len(self.optimizers)):
                self.optimizers[j].zero_grads()
            if self.is_supervised:
                inputs = [x, t]
            else:
                inputs = [x]
            self.model.train = train
            if train:
                losses = self.model(*inputs)
                if losses is None:
                    continue
                if not isinstance(losses, collections.Sequence):
                    losses = [losses]
                self.model.to_gpu()
                for j, loss in enumerate(losses):
                    loss.backward()
                    self.optimizers[j].update()
            else:
                losses = self.model(*inputs)
                if not isinstance(losses, collections.Sequence):
                    losses = [losses]
            for j, loss in enumerate(losses):
                sum_loss[j] += len(t.data) * float(loss.data)
            if self.is_supervised:
                sum_accuracy += len(t.data) * float(self.model.accuracy.data)
        y_data = self.model.y.data
        if self.on_gpu:
            x_batch = cuda.to_cpu(x_batch)
            y_data = cuda.to_cpu(y_data)
        return sum_loss, sum_accuracy, x_batch, y_data

    @staticmethod
    def dataset_to_xt_data(dataset, crop_roi):
        """Convert dataset to x_data and t_data"""
        x_data = []
        for raw_path in dataset.filenames:
            import cv2
            raw = cv2.imread(raw_path)
            raw = resize(raw, (267, 178), preserve_range=True)
            x_data.append(im_to_blob(raw))
        x_data = np.array(x_data, dtype=np.float32)
        t_data = dataset.target.astype(np.int32)
        return x_data, t_data

    def main_loop(self, n_epoch=10, save_interval=None, save_encoded=True):
        save_interval = save_interval or (n_epoch // 10)
        train_data = get_inhand_mask(which_set='train')
        test_data = get_inhand_mask(which_set='test')
        N_train = len(train_data.filenames)
        N_test = len(test_data.filenames)
        logging.info('converting dataset to x and t data')
        train_x, train_t = self.dataset_to_xt_data(train_data, self.crop_roi)
        test_x, test_t = self.dataset_to_xt_data(test_data, self.crop_roi)
        for epoch in xrange(0, n_epoch):
            # train
            sum_loss, sum_accuracy, _, _ = \
                self.batch_loop(train_x, train_t, train=True)
            for loss_id, sl in sorted(sum_loss.items()):
                mean_loss = sl / N_train
                msg = 'epoch:{:02d}; train mean loss{}={};'\
                    .format(epoch, loss_id, mean_loss)
                if self.is_supervised:
                    mean_accuracy = sum_accuracy / N_train
                    msg += ' accuracy={};'.format(mean_accuracy)
                logging.info(msg)
                print(msg)
            # test
            sum_loss, sum_accuracy, x_batch, y_batch = \
                self.batch_loop(test_x, test_t, train=False)
            for loss_id, sl in sorted(sum_loss.items()):
                mean_loss = sl / N_test
                msg = 'epoch:{:02d}; test mean loss{}={};'\
                    .format(epoch, loss_id, mean_loss)
                if self.is_supervised:
                    mean_accuracy = sum_accuracy / N_test
                    msg += ' accuracy={};'.format(mean_accuracy)
                logging.info(msg)
                print(msg)
            # save model and input/encoded/decoded
            if epoch % save_interval == (save_interval - 1):
                print('epoch:{:02d}; saving'.format(epoch))
                # save model
                model_path = osp.join(
                    self.log_dir,
                    '{name}_model_{epoch}.h5'.format(
                        name=self.model_name, epoch=epoch))
                serializers.save_hdf5(model_path, self.model)
                # save optimizer
                for i, opt in enumerate(self.optimizers):
                    opt_path = osp.join(
                        self.log_dir,
                        '{name}_optimizer_{epoch}_{i}.h5'.format(
                            name=self.model_name, epoch=epoch, i=i))
                    serializers.save_hdf5(opt_path, opt)
                # save x_data
                x_path = osp.join(self.log_dir, 'x_{}.pkl'.format(epoch))
                with open(x_path, 'wb') as f:
                    pickle.dump(x_batch, f)  # save x
                if not self.is_supervised:
                    x_hat_path = osp.join(self.log_dir,
                                          'x_hat_{}.pkl'.format(epoch))
                    with open(x_hat_path, 'wb') as f:
                        pickle.dump(y_batch, f)  # save x_hat
                    tile_ae_inout(
                        x_batch, y_batch,
                        osp.join(self.log_dir, 'X_{}.jpg'.format(epoch)))
                if save_encoded:
                    x = Variable(cuda.to_gpu(x_batch), volatile=True)
                    z = self.model.encode(x)
                    tile_ae_encoded(
                        cuda.to_cpu(z.data),
                        osp.join(self.log_dir,
                                 'x_encoded_{}.jpg'.format(epoch)))

        for i in xrange(len(self.optimizers)):
            draw_loss_curve(
                loss_id=i,
                logfile=self.log_file,
                outfile=osp.join(self.log_dir, 'loss_curve{}.jpg'.format(i)),
                no_acc=not self.is_supervised,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'supervised_or_not', type=str, choices=['supervised', 'unsupervised'],
        help='do supervised or unsupervised training')
    parser.add_argument('--epoch', type=int, default=50,
                        help='number of recursion (default: 50)')
    parser.add_argument('--no-logging', action='store_true',
                        help='logging to tmp dir')
    parser.add_argument('--save-interval', type=int, default=None,
                        help='save interval of x and x_hat')
    parser.add_argument('-m', '--model', required=True, help='name of model')
    args = parser.parse_args()

    n_epoch = args.epoch
    save_interval = args.save_interval
    is_supervised = True if args.supervised_or_not == 'supervised' else False

    on_gpu = True
    is_pipeline = False
    batch_size = 10
    save_encoded = False
    crop_roi = False
    optimizers = [O.Adam()]
    if is_supervised:
        if args.model == 'VGG_mini_ABN':
            from apc_od.models import VGG_mini_ABN
            model = VGG_mini_ABN()
            if on_gpu:
                model.to_gpu()
            optimizers[0].setup(model)
            crop_roi = True
        elif args.model == 'CAEOnesRoiVGG':
            from apc_od.pipeline import CAEOnesRoiVGG
            is_pipeline = True
            batch_size = 10
            initial_roi = np.array([100, 130, 300, 400])
            initial_roi = roi_preprocess(initial_roi)
            # setup model
            model = CAEOnesRoiVGG(initial_roi)
            if on_gpu:
                model.to_gpu()
            optimizers = [O.Adam(), O.Adam()]
            optimizers[0].setup(model.cae_ones1)
            optimizers[1].setup(model.vgg2)
            # load trained models
            serializers.load_hdf5(os.path.join(here, 'cae_ones_model.h5'),
                                  model.cae_ones1)
            serializers.load_hdf5(os.path.join(here, 'vgg_model.h5'),
                                  model.vgg2)
            # load optimizers state
            serializers.load_hdf5(os.path.join(here, 'cae_ones_optimizer.h5'),
                                  optimizers[0])
            serializers.load_hdf5(os.path.join(here, 'vgg_optimizer.h5'),
                                  optimizers[1])
        else:
            sys.stderr.write('Unsupported model: {}\n'.format(args.model))
            sys.exit(1)
    else:
        # unsupervised
        if args.model == 'CAE':
            from apc_od.models import CAE
            save_encoded = True
            model = CAE()
            if on_gpu:
                model.to_gpu()
            optimizers[0].setup(model)
        elif args.model == 'CAEOnes':
            from apc_od.models import CAEOnes
            model = CAEOnes(n_param=1)
            if on_gpu:
                model.to_gpu()
            optimizers[0].setup(model)
        elif args.model == 'CAEPool':
            from apc_od.models import CAEPool
            save_encoded = True
            model = CAEPool()
            if on_gpu:
                model.to_gpu()
            optimizers[0].setup(model)
        elif args.model == 'StackedCAE':
            from apc_od.models import StackedCAE
            save_encoded = True
            model = StackedCAE()
            if on_gpu:
                model.to_gpu()
            optimizers[0].setup(model)
        else:
            sys.stderr.write('Unsupported model: {}\n'.format(args.model))
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

    trainer = Trainer(
        optimizers=optimizers,
        model=model,
        model_name=args.model,
        is_supervised=is_supervised,
        crop_roi=crop_roi,
        batch_size=batch_size,
        log_dir=log_dir,
        log_file=log_file,
        on_gpu=on_gpu,
    )
    trainer.main_loop(
        n_epoch=n_epoch,
        save_interval=save_interval,
        save_encoded=save_encoded,
    )


if __name__ == '__main__':
    main()
