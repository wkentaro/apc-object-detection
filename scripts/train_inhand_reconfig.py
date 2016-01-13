#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle as pickle
import gzip
import os
import logging

from chainer import Variable
import chainer.functions as F
import cv2
import numpy as np
from sklearn.datasets import load_files
from sklearn.preprocessing import normalize
from skimage.morphology import closing, square

from jsk_recognition_utils import bounding_rect_of_mask
from imagesift import get_sift_keypoints
from jsk_apc2015_common import object_list

logging.basicConfig(
    filename='log_train_inhand.txt',
    level=logging.DEBUG,
)


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
    img = cv2.imread(img_file)
    mask = cv2.imread(mask_file)
    train_imgs.append((img, mask))
test_imgs = []
for f in test_data.filenames:
    if f.endswith('_0.jpg'):
        # Skip mask file
        continue
    img_file = f
    mask_file = img_file.split('_1.jpg')[0] + '_0.jpg'
    img = cv2.imread(img_file)
    mask = cv2.imread(mask_file)
    test_imgs.append((img, mask))

object_list = np.array(object_list)
print(list(object_list).index('kong_sitting_frog_dog_toy'))

y_true = np.zeros(25, dtype=np.float32)
y_true[12] = 1

learning_n_sample = 50
learning_rate = 0.5
initial_params = [8, 4]  # size, iterations
for epoch in xrange(100):
    # train
    sum_loss = 0
    for img, mask in train_imgs:
        mask = resize(mask, (267, 178), preserve_range=True)
        x_data = np.array([apc_od.im_to_blob(mask)])
        x = Variable(x_data, volatile='on')
        z = model.encode(x)
        param_scale = z.data
        rands_shape = [learning_n_sample] + list(param_scale.shape)
        rands = learning_rate * (2 * np.random.random(rands_shape) - 1) + 1
        rands[0] = np.ones(param_scale.shape)  # ones
        min_rand = None
        min_error = np.inf
        for rand in rands:
            params = rand * param_scale * initial_params
            size, iterations = map(int, params)
            kernel = np.ones((size, size), dtype=np.uint8)
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            cropped = bounding_rect_of_mask(img, closed)
            frames, desc = get_sift_keypoints(cropped)
            X = np.array([desc])
            X_trans = bof.transform(X)
            X_trans = normalize(X_trans)
            y_proba = lgr.predict_proba(X_trans)[0]
            square_error = np.power(y_proba - y_true, 2)
            if square_error < min_error:
                min_error = square_error
                min_rand = rand
        t_data = np.array(min_rand * param_scale * initial_params, dtype=np.float32)
        t = Variable(t_data, volatile='off')
        loss = F.mean_squared_error(t, z)
        loss.backward()
        sum_loss += float(loss.data)
    mean_loss = 1. * sum_loss / len(train_imgs)
    msg = 'epoch:{:02d}; train mean loss0={};'.format(epoch, mean_loss)
    logging.info(msg)
    print(msg)

    # test
    sum_loss = 0
    for img, mask in test_imgs:
        mask = resize(mask, (267, 178), preserve_range=True)
        x_data = np.array([apc_od.im_to_blob(mask)])
        x = Variable(x_data, volatile='on')
        z = model.encode(x)
        param_scale = z.data
        params = np.array(param_scale * initial_params, dtype=np.float32)
        size, iterations = map(int, params)
        kernel = np.ones((size, size), dtype=np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        cropped = bounding_rect_of_mask(img, closed)
        frames, desc = get_sift_keypoints(cropped)
        X = np.array([desc])
        X_trans = bof.transform(X)
        X_trans = normalize(X_trans)
        y_proba = lgr.predict_proba(X_trans)[0]
        square_error = np.power(y_proba - y_true, 2)
        sum_loss += square_error
    mean_loss = 1. * sum_loss / len(test_imgs)
    msg = 'epoch:{:02d}; test mean loss0={};'.format(epoch, mean_loss)
    logging.info(msg)
    print(msg)
