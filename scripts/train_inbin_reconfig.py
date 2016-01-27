#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import argparse
import glob
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
from pcl_ros.srv import UpdateFilename
from pcl_ros.srv import UpdateFilenameRequest

import apc_od
from apc_od.models import CAEOnes

log_file = 'log_inbin_reconfig.txt'

def write_log(msg):
    with open(log_file, 'a') as f:
        f.write(msg + '\n')


def save_label_and_img(fname, label, img):
    from skimage.color import label2rgb
    from skimage.io import imsave
    rgb = label2rgb(label=label, image=img, bg_label=0)
    imsave(fname, rgb)


class TrainInBinReconfig(object):

    def __init__(self, model, optimizer, x, mask, image, all_mask):
        self.model = model
        self.optimizer = optimizer
        self.x = x
        self.mask = mask
        self.image = image
        self.all_mask = (all_mask > 127).astype(int)
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
        val = self.evaluate_label(label, self.mask)
        return val, label

    def evaluate_label(self, label, mask):
        """Evaluate diff between label and mask"""
        unique_labels = np.unique(label)
        min_val = np.inf
        mask = (mask > 127).astype(int)
        for l in unique_labels:
            label_mask = (label == l).astype(int)
            val = 1. * np.sum(np.abs(label_mask - mask)) / self.all_mask.sum()
            if val < min_val:
                min_val = val
        return min_val

    def random_sample(self, fname, epoch):
        learning_rate = 1
        learning_n_sample = 30

        self.model.train = True
        z = self.model.encode(self.x)
        param_scale = z.data

        self.optimizer.zero_grads()
        rands_shape = [learning_n_sample] + list(param_scale.shape)
        rands = 1. * learning_rate * (21 - epoch) / 20 * (2 * np.random.random(rands_shape) - 1) + 1
        rands[0] = np.ones(param_scale.shape)  # ones
        min_label = None
        min_value = np.inf
        min_rand = None
        for i, rand in enumerate(rands):
            param_scale_with_rand = rand * param_scale
            params = (self.initial_param * param_scale_with_rand)[0]
            if params[0] < 0.0009:
                min_rand = 1. / param_scale
                params[0] = self.initial_param[0]
                self.reconfigure(tolerance=params[0])
                min_value, min_label = self.evaluate()
                if params[0] < 0:
                    break
                else:
                    continue
            self.reconfigure(tolerance=params[0])
            val, label = self.evaluate()
            if val < min_value:
                min_label = label
                min_value = val
                min_rand = rand
        if min_rand is None:
            return np.inf, -np.inf
        save_label_and_img(fname, min_label, self.image)
        t_data = (min_rand * param_scale).astype(np.float32)
        print('unique_labels: {}, param_scale: {} -> t_data: {}, min_value: {}'.format(np.unique(min_label), param_scale, t_data, min_value))
        t = Variable(t_data, volatile='off')
        loss = F.mean_squared_error(z, t)
        loss.backward()
        self.optimizer.update()
        accuracy = 1. - min_value
        return float(loss.data), accuracy


def main():
    rospy.init_node('train_inbin_reconfig')

    update_filename_client = rospy.ServiceProxy(
        'pcd_to_pointcloud/update_filename', UpdateFilename)

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

    # test
    epoch = 0
    sum_acc = 0
    for i in [4, 5]:
        pcd_file = os.path.realpath(glob.glob('{}/{}/*.pcd'.format(directory, i))[0])
        req = UpdateFilenameRequest(filename=pcd_file)
        res = update_filename_client(req)
        print('update_filename to: {}, success: {}'
              .format(pcd_file, res.success))

        image_file = os.path.join(directory, str(i), 'image.jpg')
        depth_file = os.path.join(directory, str(i), 'depth.jpg')
        all_mask_file = os.path.join(directory, str(i), 'mask.jpg')
        diffmask_file = os.path.join(directory, str(i), 'diffmask2.jpg')
        image = cv2.imread(image_file)
        depth = cv2.imread(depth_file)
        all_mask = cv2.imread(all_mask_file, 0)
        mask = cv2.imread(diffmask_file, 0)
        depth = resize(depth, (267, 178), preserve_range=True)
        x_data = np.array([apc_od.im_to_blob(depth)], dtype=np.float32)
        x = Variable(x_data, volatile='on')
        trainer = TrainInBinReconfig(model=model, optimizer=optimizer,
                                     x=x, mask=mask, image=image, all_mask=all_mask)
        model.train = False
        param_scale = model.encode(x).data
        params = (trainer.initial_param * param_scale)[0]
        tolerance = params[0]
        if params[0] < 0.0009:
            params[0] = trainer.initial_param[0]
            trainer.reconfigure(tolerance=params[0])
        trainer.reconfigure(tolerance=tolerance)
        val, label = trainer.evaluate()
        save_label_and_img('{}_{}.jpg'.format(i, epoch), label, image)
        sum_acc += (1. - val)
    mean_acc = sum_acc / 2.
    msg = 'epoch:{:02d}; test mean accuracy0={};'.format(epoch, mean_acc)
    write_log(msg)

    for epoch in xrange(1, 21):
        sum_loss = 0
        sum_acc = 0
        for i in [1, 2, 3]:
            pcd_file = os.path.realpath(glob.glob('{}/{}/*.pcd'.format(directory, i))[0])
            req = UpdateFilenameRequest(filename=pcd_file)
            res = update_filename_client(req)
            print('update_filename to: {}, success: {}'
                  .format(pcd_file, res.success))

            image_file = os.path.join(directory, str(i), 'image.jpg')
            depth_file = os.path.join(directory, str(i), 'depth.jpg')
            all_mask_file = os.path.join(directory, str(i), 'mask.jpg')
            diffmask_file = os.path.join(directory, str(i), 'diffmask2.jpg')

            image = cv2.imread(image_file)
            depth = cv2.imread(depth_file)
            all_mask = cv2.imread(all_mask_file, 0)
            mask = cv2.imread(diffmask_file, 0)
            depth = resize(depth, (267, 178), preserve_range=True)
            x_data = np.array([apc_od.im_to_blob(depth)], dtype=np.float32)
            x = Variable(x_data, volatile='off')

            trainer = TrainInBinReconfig(
                model=model, optimizer=optimizer, x=x, mask=mask, image=image, all_mask=all_mask)
            model.train = True
            loss_data, acc = trainer.random_sample(fname='{}_{}.jpg'.format(i, epoch), epoch=epoch)
            sum_loss += loss_data
            sum_acc += acc
        mean_loss = sum_loss / 3.
        mean_acc = sum_acc / 3.
        msg = 'epoch:{:02d}; train mean loss0={};'.format(epoch, mean_loss)
        write_log(msg)
        msg = 'epoch:{:02d}; train mean accuracy0={};'.format(epoch, mean_acc)
        write_log(msg)

        # test
        sum_acc = 0
        for i in [4, 5]:
            pcd_file = os.path.realpath(glob.glob('{}/{}/*.pcd'.format(directory, i))[0])
            req = UpdateFilenameRequest(filename=pcd_file)
            res = update_filename_client(req)
            print('update_filename to: {}, success: {}'
                  .format(pcd_file, res.success))

            image_file = os.path.join(directory, str(i), 'image.jpg')
            depth_file = os.path.join(directory, str(i), 'depth.jpg')
            all_mask_file = os.path.join(directory, str(i), 'mask.jpg')
            diffmask_file = os.path.join(directory, str(i), 'diffmask2.jpg')
            image = cv2.imread(image_file)
            depth = cv2.imread(depth_file)
            all_mask = cv2.imread(all_mask_file, 0)
            mask = cv2.imread(diffmask_file, 0)
            depth = resize(depth, (267, 178), preserve_range=True)
            x_data = np.array([apc_od.im_to_blob(depth)], dtype=np.float32)
            x = Variable(x_data, volatile='on')
            trainer = TrainInBinReconfig(model=model, optimizer=optimizer,
                                        x=x, mask=mask, image=image, all_mask=all_mask)
            model.train = False
            param_scale = model.encode(x).data
            params = (trainer.initial_param * param_scale)[0]
            if params[0] < 0.0009:
                params[0] = trainer.initial_param[0]
            tolerance = params[0]
            trainer.reconfigure(tolerance=tolerance)
            val, label = trainer.evaluate()
            save_label_and_img('{}_{}.jpg'.format(i, epoch), label, image)
            sum_acc += (1. - val)
        mean_acc = sum_acc / 2.
        msg = 'epoch:{:02d}; test mean accuracy0={};'.format(epoch, mean_acc)
        write_log(msg)

    S.save_hdf5('cae_ones_model_inbin_trained.h5', model)
    S.save_hdf5('cae_ones_optimizer_inbin_trained.h5', optimizer)


if __name__ == '__main__':
    main()
