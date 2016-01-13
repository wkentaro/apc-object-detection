#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import argparse
import os
import logging

from chainer import functions as F
from chainer import optimizers as O
from chainer import serializers as S
from chainer import Variable
import cv2
import numpy as np
from skimage.transform import resize

import cv_bridge
import dynamic_reconfigure.client
import rospy
from sensor_msgs.msg import Image

import apc_od
from apc_od.models import CAEOnes


log_file = 'log.txt'


def write_log(msg):
    with open(log_file, 'a') as f:
        f.write(msg)


class TrainInBinReconfig(object):

    def __init__(self, model, optimizer, x, mask):
        self.model = model
        self.optimizer = optimizer
        self.x = x
        self.mask = mask
        self.initial_param = np.array([0.02])

        self.label_msg = None
        self.reconfig_client = \
            dynamic_reconfigure.client.Client('euclid_cluster')

        self.sub_label = rospy.Subscriber('cpi_decomposer/label',
                                          Image, self._label_cb)

    def _label_cb(self, msg):
        self.label_msg = msg

    def reconfigure(self, tolerance):
        params = {'tolerance': tolerance}
        self.reconfig_client.update_configuration(params)

    def evaluate(self):
        """Wait for reconfigured result and return the evaluation value"""
        now = rospy.Time.now()
        while self.label_msg is None:
            rospy.sleep(0.1)
        while self.label_msg.header.stamp < now:
            rospy.sleep(0.1)
        label_msg = self.label_msg
        bridge = cv_bridge.CvBridge()
        label = bridge.imgmsg_to_cv2(label_msg)
        val = self.evaluate_label(label, mask)
        return val

    def evaluate_label(self, label, mask, bg_label=0):
        """Evaluate diff between label and mask"""
        unique_labels = np.unique(label)
        print(unique_labels)
        min_val = np.inf
        mask = (mask > 127).astype(int)
        img_size = mask.size
        for l in unique_labels:
            if l == bg_label:
                continue
            label_mask = (label == l).astype(int)
            val = 1. * np.sum(np.abs(label_mask - mask)) / img_size
            print(val)
            if val < min_val:
                min_val = val
        return min_val

    def random_sample(self):
        learning_rate = 1
        learning_n_sample = 10

        self.model.train = True
        z = self.model.encode(self.x)
        param_scale = z.data

        rands_shape = [learning_n_sample] + list(param_scale.shape)
        rands = learning_rate * (2 * np.random.random(rands_shape) - 1) + 1
        rands[0] = np.ones(param_scale.shape)  # ones
        min_value = np.inf
        for i, rand in enumerate(rands):
            param_scale_with_rand = rand * param_scale
            params = (self.initial_param * param_scale_with_rand)[0]
            print('tolerance: ', params[0])
            self.reconfigure(tolerance=params[0])
            val = self.evaluate()
            print('eval value: ', val)
            if val < min_value:
                min_value = val
                min_rand = rand
        t_data = (min_rand * param_scale).astype(np.float32)
        print('t_data: ', t_data)
        t = Variable(t_data, volatile='off')
        loss = F.mean_squared_error(z, t)
        loss.backward()
        return float(loss.data)


if __name__ == '__main__':
    rospy.init_node('train_inbin_reconfig')

    parser = argparse.ArgumentParser()
    parser.add_argument('container_dir')
    args = parser.parse_args()

    model = CAEOnes(n_param=1)
    optimizer = O.Adam()
    optimizer.setup(model)
    print('loading hd5')
    S.load_hdf5('cae_ones_model_inbin.h5', model)
    S.load_hdf5('cae_ones_optimizer_inbin.h5', optimizer)
    print('done loading')

    directory = args.container_dir
    import datetime as dt
    for epoch in xrange(10):
        sum_loss = 0
        for i in xrange(1, 5):
            while True:
                yn = raw_input('Next is {}, can go? [y/n]: '.format(i))
                if yn.lower() == 'y':
                    break
            depth_file = os.path.join(directory, str(i), 'depth.jpg')
            diffmask_file = os.path.join(directory, str(i), 'diffmask2.jpg')

            depth = cv2.imread(depth_file)
            mask = cv2.imread(diffmask_file, 0)
            depth = resize(depth, (267, 178), preserve_range=True)
            x_data = np.array([apc_od.im_to_blob(depth)], dtype=np.float32)
            x = Variable(x_data, volatile='off')

            trainer = TrainInBinReconfig(
                model=model, optimizer=optimizer, x=x, mask=mask)
            loss_data = trainer.random_sample()
            sum_loss += loss_data
        mean_loss = 1. * sum_loss / 5
        msg = 'epoch:{:02d}; train mean loss0={};'.format(epoch, mean_loss)
        write_log(msg)

        # test
        while True:
            yn = raw_input('Next is {}, can go? [y/n]: '.format(5))
            if yn.lower() == 'y':
                break
        depth_file = os.path.join(directory, str(5), 'depth.jpg')
        diffmask_file = os.path.join(directory, str(5), 'diffmask.jpg')
        depth = cv2.imread(depth_file)
        mask = cv2.imread(diffmask_file, 0)
        depth = resize(depth, (267, 178), preserve_range=True)
        x_data = np.array([apc_od.im_to_blob(depth)], dtype=np.float32)
        x = Variable(x_data, volatile='on')
        trainer = TrainInBinReconfig(model=model, optimizer=optimizer,
                                     x=x, mask=mask)
        model.train = False
        param_scale = model.encode(x).data
        params = (trainer.initial_param * param_scale)[0]
        tolerance = params[0]
        trainer.reconfigure(tolerance=tolerance)
        val = trainer.evaluate()
        msg = 'epoch:{:02d}; test mean loss0={};'.format(epoch, val)
        write_log(msg)

    S.save_hdf5('cae_ones_model_inbin_trained.h5', model)
    S.save_hdf5('cae_ones_optimizer_inbin_trained.h5', optimizer)
