#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer.serializers as S
from chainer import Variable

import apc_od
from apc_od.models import CAEOnes
import numpy as np
from skimage.transform import resize

import cv_bridge
import dynamic_reconfigure.client
import rospy
from sensor_msgs.msg import Image


class CAENode(object):

    def __init__(self, model):
        self.model = model
        self.sub = rospy.Subscriber('~input/depth', Image, self.cb)
        self.reconfig_client = \
            dynamic_reconfigure.client.Client('bin_e_euclid_clustering')

    def cb(self, msg):
        bridge = cv_bridge.CvBridge()
        depth_tmp = bridge.imgmsg_to_cv2(msg)
        shape = list(depth_tmp.shape)
        shape.append(3)
        depth = np.zeros(shape, depth_tmp.dtype)
        depth[:,:,0] = depth_tmp
        depth[:,:,1] = depth_tmp
        depth[:,:,2] = depth_tmp
        depth = depth.astype(float)
        depth = depth / depth.max() * 255
        depth = depth.astype(np.uint8)
        depth = resize(depth, (267, 178), preserve_range=True)
        x_data = np.array([apc_od.im_to_blob(depth)], dtype=np.float32)
        x = Variable(x_data, volatile='off')
        z = self.model.encode(x)
        initial_param = np.array([0.02])
        scale = z.data
        tolerance = (initial_param * scale)[0][0]
        params = {'tolerance': tolerance}
        self.reconfig_client.update_configuration(params)


def main():
    rospy.init_node('cae_param_node')

    model = CAEOnes(n_param=1)
    S.load_hdf5('cae_ones_model_inbin_trained.h5', model)

    model.train = False
    CAENode(model)

    rospy.spin()


if __name__ == '__main__':
    main()
