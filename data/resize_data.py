#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from skimage.io import imsave
from skimage.transform import resize

import apc_od


def main():
    raw_files = apc_od.get_original('raw').filenames
    mask_files = apc_od.get_original('mask').filenames
    for rf, mf in zip(raw_files, mask_files):
        raw_img = cv2.imread(rf)
        raw_img = resize(raw_img, (712, 1068))
        imsave(rf, raw_img)
        mask_img = cv2.imread(mf)
        mask_img = resize(mask_img, (712, 1068))
        imsave(mf, mask_img)


if __name__ == '__main__':
    main()

