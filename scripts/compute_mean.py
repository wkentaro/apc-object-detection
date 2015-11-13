#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os.path as osp

import cv2
import numpy as np
from sklearn.datasets import load_files

import apc_od


def compute_mean_img(imgs):
    return np.mean(imgs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('container_path', help='dir contains images')
    parser.add_argument('-O', '--output', help='output file')
    args = parser.parse_args()

    container_path = args.container_path
    output = args.output or osp.basename(container_path) + '_mean_img.jpg'

    files = load_files(container_path, load_content=False).filenames
    imgs = np.array([cv2.imread(f) for f in files])
    mean_img = compute_mean_img(imgs)
    assert mean_img.shape == imgs[0].shape

    cv2.imwrite(output, mean_img)


if __name__ == '__main__':
    main()