#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle as pickle
import gzip
import os

from chainer import Variable
import chainer.functions as F
import chainer.optimizers as O
import chainer.serializers as S
import cv2
import numpy as np
from sklearn.datasets import load_files
from sklearn.preprocessing import normalize
from skimage.morphology import closing, square
from skimage.transform import resize

from jsk_recognition_utils import bounding_rect_of_mask
from imagesift import get_sift_keypoints


def write_log(msg):
    log_file = 'log_train_inhand.txt'
    with open(log_file, 'a') as f:
        f.write(msg + '\n')


import apc_od
from apc_od.models import CAEOnes


def main():
    model = CAEOnes(n_param=2)
    optimizer = O.Adam()
    optimizer.setup(model)
    S.load_hdf5('bof_data/cae_ones_model.h5', model)
    S.load_hdf5('bof_data/cae_ones_optimizer.h5', optimizer)

    with gzip.open('bof_data/bof_berkeley.pkl.gz', 'rb') as f:
        bof = pickle.load(f)
    with gzip.open('bof_data/lgr_merged.pkl.gz', 'rb') as f:
        lgr = pickle.load(f)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_home = os.path.abspath(os.path.join(this_dir, '../data'))
    train_data_dir = os.path.join(data_home, 'in_hand_recog_{}'.format('train'))
    test_data_dir = os.path.join(data_home, 'in_hand_recog_{}'.format('test'))

    train_data = load_files(train_data_dir, load_content=False)
    test_data = load_files(test_data_dir, load_content=False)

    N_train = len(train_data.filenames)
    N_test = len(test_data.filenames)

    train_imgs = []
    for f in train_data.filenames:
        if f.endswith('_0.jpg'):
            # Skip mask file
            continue
        img_file = f
        mask_file = img_file.split('_1.jpg')[0] + '_0.jpg'
        img = cv2.imread(img_file, 0)
        mask = cv2.imread(mask_file)
        train_imgs.append((img, mask))
    test_imgs = []
    for f in test_data.filenames:
        if f.endswith('_0.jpg'):
            # Skip mask file
            continue
        img_file = f
        mask_file = img_file.split('_1.jpg')[0] + '_0.jpg'
        img = cv2.imread(img_file, 0)
        mask = cv2.imread(mask_file)
        test_imgs.append((img, mask))

    y_true = np.zeros(25, dtype=np.float32)
    y_true[12] = 1




    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # test
    n_batch = 10
    initial_params = [8, 4]  # size, iterations
    epoch = 0
    model.train = False
    sum_error = 0
    print('testing')
    N = len(test_imgs)
    perm = np.random.permutation(len(test_imgs))
    for i in xrange(0, N, n_batch):
        print('test_batch: ', i)
        test_batch = [test_imgs[p_index] for p_index in perm[i:i+n_batch]]
        x_data = []
        for img, mask in test_batch:
            mask = resize(mask, (267, 178), preserve_range=True)
            x_data.append(apc_od.im_to_blob(mask))
        x_data = np.array(x_data, dtype=np.float32)
        x = Variable(x_data, volatile='on')
        z = model.encode(x)
        param_scale = z.data
        params = param_scale * initial_params

        X = []
        for k, param in enumerate(params):
            size, iterations = map(int, param)
            if size <= 0 or size > 50 or iterations <= 0 or iterations > 50:
                rand = 1. * np.ones(2) / param_scale
                params = rand * param_scale * initial_params
                size, iterations = map(int, params[0])
                print('test:', size, iterations)
            if size <= 0 or size > 50 or iterations <= 0 or iterations > 50:
                size, iterations = initial_params
            kernel = np.ones((size, size), dtype=np.uint8)
            img, mask = test_batch[k]
            closed = cv2.morphologyEx(mask[:,:,0], cv2.MORPH_CLOSE, kernel, iterations=iterations)
            cropped = bounding_rect_of_mask(img, closed)
            frames, desc = get_sift_keypoints(cropped)
            X.append(desc)
        X = np.array(X)
        if X.size == 0:
            print('test: skipping')
            N -= n_batch
            continue

        X_trans = bof.transform(X)
        X_trans = normalize(X_trans)
        y_proba = lgr.predict_proba(X_trans)[0]
        square_error = np.sum(np.power(y_proba - y_true, 2))
        sum_error += square_error
    try:
        mean_error = 1. * sum_error / N
    except ZeroDivisionError:
        mean_error = np.inf
    msg = 'epoch:{:02d}; test mean loss1={};'.format(epoch, mean_error)
    write_log(msg)
    print(msg)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




    n_batch = 10
    learning_n_sample = 100
    learning_rate = 0.1
    initial_params = [8, 4]  # size, iterations
    for epoch in xrange(1, 11):
        print('epoch:', epoch)
        # train
        model.train = True
        sum_loss = 0
        sum_error = 0
        N = len(train_imgs)
        N_train = len(train_imgs)
        perm = np.random.permutation(N_train)
        for i in range(0, N_train, n_batch):
            print('train_batch: ', i)
            train_batch = [train_imgs[p_index] for p_index in perm[i:i+n_batch]]
            x_data = []
            for img, mask in train_batch:
                mask = resize(mask, (267, 178), preserve_range=True)
                x_data.append(apc_od.im_to_blob(mask))
            x_data = np.array(x_data, dtype=np.float32)
            x = Variable(x_data, volatile='off')
            z = model.encode(x)
            param_scale = z.data
            rands_shape = [learning_n_sample] + list(param_scale.shape)
            rands = 1. * learning_rate * (11 - epoch) / 11 * (2 * np.random.random(rands_shape) - 1) + 1
            rands[0] = np.ones(param_scale.shape)  # ones
            min_rand = None
            min_error = np.inf
            optimizer.zero_grads()
            for j, rand in enumerate(rands):
                params = rand * param_scale * initial_params
                X = []
                for k, param in enumerate(params):
                    size, iterations = map(int, param)
                    if size <= 0 or size > 50 or iterations <= 0 or iterations > 50:
                        size, iterations = initial_params
                    kernel = np.ones((size, size), dtype=np.uint8)
                    img, mask = train_batch[k]
                    closed = cv2.morphologyEx(mask[:,:,0], cv2.MORPH_CLOSE, kernel, iterations=iterations)
                    cropped = bounding_rect_of_mask(img, closed)
                    frames, desc = get_sift_keypoints(cropped)
                    X.append(desc)
                X = np.array(X)
                if X.size == 0:
                    continue
                X_trans = bof.transform(X)
                X_trans = normalize(X_trans)
                y_proba = lgr.predict_proba(X_trans)[0]
                square_error = 1. * np.sum(np.power(y_proba - y_true, 2)) / len(train_batch)
                if square_error < min_error:
                    min_error = square_error
                    min_rand = rand
            if min_rand is None:
                print('train: skipping')
                N -= n_batch
                continue
            t_data = np.array(min_rand * param_scale, dtype=np.float32)
            t = Variable(t_data, volatile='off')
            loss = F.mean_squared_error(t, z)
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(train_batch)
            sum_error += min_error * len(train_batch)
        try:
            mean_loss = 1. * sum_loss / N
        except ZeroDivisionError:
            mean_loss = np.inf
        mean_error = 1. * sum_error / N
        msg = 'epoch:{:02d}; train mean loss0={};'.format(epoch, mean_loss)
        write_log(msg)
        print(msg)
        msg = 'epoch:{:02d}; train mean loss1={};'.format(epoch, mean_error)
        write_log(msg)
        print(msg)

        # test
        model.train = False
        sum_error = 0
        print('testing')
        N = len(test_imgs)
        perm = np.random.permutation(len(test_imgs))
        for i in xrange(0, N, n_batch):
            print('test_batch: ', i)
            test_batch = [test_imgs[p_index] for p_index in perm[i:i+n_batch]]
            x_data = []
            for img, mask in test_batch:
                mask = resize(mask, (267, 178), preserve_range=True)
                x_data.append(apc_od.im_to_blob(mask))
            x_data = np.array(x_data, dtype=np.float32)
            x = Variable(x_data, volatile='on')
            z = model.encode(x)
            param_scale = z.data
            params = param_scale * initial_params

            X = []
            for k, param in enumerate(params):
                size, iterations = map(int, param)
                if size <= 0 or size > 50 or iterations <= 0 or iterations > 50:
                    rand = 1. * np.ones(2) / param_scale
                    params = rand * param_scale * initial_params
                    size, iterations = map(int, params[0])
                    print('test:', size, iterations)
                if size <= 0 or size > 50 or iterations <= 0 or iterations > 50:
                    size, iterations = initial_params
                kernel = np.ones((size, size), dtype=np.uint8)
                img, mask = test_batch[k]
                closed = cv2.morphologyEx(mask[:,:,0], cv2.MORPH_CLOSE, kernel, iterations=iterations)
                cropped = bounding_rect_of_mask(img, closed)
                frames, desc = get_sift_keypoints(cropped)
                X.append(desc)
            X = np.array(X)
            if X.size == 0:
                print('test: skipping')
                N -= n_batch
                continue

            X_trans = bof.transform(X)
            X_trans = normalize(X_trans)
            y_proba = lgr.predict_proba(X_trans)[0]
            square_error = np.sum(np.power(y_proba - y_true, 2))
            sum_error += square_error
        try:
            mean_error = 1. * sum_error / N
        except ZeroDivisionError:
            mean_error = np.inf
        msg = 'epoch:{:02d}; test mean loss1={};'.format(epoch, mean_error)
        write_log(msg)
        print(msg)

        S.save_hdf5('bof_data/cae_ones_model_trained_{}.h5'.format(epoch), model)
        S.save_hdf5('bof_data/cae_ones_optimizer_trained_{}.h5'.format(epoch), optimizer)


if __name__ == '__main__':
    main()
