#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave

from apc_od.data import blob_to_im
from apc_od.imaging import tile_slices_to_image
from apc_od.imaging import tile_slices_to_image_uint8
from apc_od.utils import atleast_4d


def tile_ae_inout(x, x_hat, output_file):
    X = np.vstack((x, x_hat))

    # tile images
    imgs = np.array([blob_to_im(xi) for xi in X])
    tiled_img = tile_slices_to_image_uint8(imgs)
    tiled_img = np.array(tiled_img)  # PIL image -> numpy.ndarray

    # save tiled image
    imsave(output_file, tiled_img)


def tile_ae_encoded(z_data, filename):
    z_data = atleast_4d(z_data)
    for i, zi in enumerate(z_data):
        tile_img = np.array(tile_slices_to_image(zi))
        base, ext = os.path.splitext(filename)
        filename_ = '{base}_{id}{ext}'.format(base=base, id=i, ext=ext)
        imsave(filename_, tile_img)


def draw_loss_curve(logfile, outfile, no_acc):
    train_loss = []
    test_loss = []
    if not no_acc:
        train_acc = []
        test_acc = []
    for line in open(logfile):
        line = line.strip()
        if 'epoch:' not in line:
            continue
        epoch = int(re.search('epoch:([0-9]+?);', line).groups()[0])
        if 'train' in line:
            tr_l = float(re.search('loss=(.+?);', line).groups()[0])
            train_loss.append([epoch, tr_l])
            if not no_acc:
                tr_a = float(re.search('accuracy=([0-9\.]+?);', line)
                             .groups()[0])
                train_acc.append([epoch, tr_a])
        if 'test' in line:
            te_l = float(re.search('loss=(.+?);', line).groups()[0])
            test_loss.append([epoch, te_l])
            if not no_acc:
                te_a = float(re.search('accuracy=([0-9\.]+?);', line)
                             .groups()[0])
                test_acc.append([epoch, te_a])

    train_loss = np.asarray(train_loss)
    test_loss = np.asarray(test_loss)
    if not no_acc:
        train_acc = np.asarray(train_acc)
        test_acc = np.asarray(test_acc)

    if not len(train_loss) > 2:
        return

    fig, ax1 = plt.subplots()
    ax1.plot(train_loss[:, 0], train_loss[:, 1], label='training loss')
    ax1.plot(test_loss[:, 0], test_loss[:, 1], label='test loss')
    ax1.set_xlim([1, len(train_loss)])
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')

    if not no_acc:
        ax2 = ax1.twinx()
        ax2.plot(train_acc[:, 0], train_acc[:, 1],
                 label='training accuracy', c='r')
        ax2.plot(test_acc[:, 0], test_acc[:, 1], label='test accuracy', c='c')
        ax2.set_xlim([1, len(train_loss)])
        ax2.set_ylabel('accuracy')

    ax1.legend(bbox_to_anchor=(0.25, -0.1), loc=9)
    if not no_acc:
        ax2.legend(bbox_to_anchor=(0.75, -0.1), loc=9)
    plt.savefig(outfile, bbox_inches='tight')
