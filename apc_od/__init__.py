#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')

import apc_od.data
import apc_od.dataset
import apc_od.draw
import apc_od.imaging


__version__ = '0.1'


im_preprocess = apc_od.data.im_preprocess
im_to_blob = apc_od.data.im_to_blob
blob_to_im = apc_od.data.blob_to_im
mask_to_roi = apc_od.data.mask_to_roi
roi_preprocess = apc_od.data.roi_preprocess


get_raw = apc_od.dataset.get_raw
get_inbin_depth = apc_od.dataset.get_inbin_depth
raw_to_mask_path = apc_od.dataset.raw_to_mask_path


draw_loss_curve = apc_od.draw.draw_loss_curve
tile_ae_encoded = apc_od.draw.tile_ae_encoded
tile_ae_inout = apc_od.draw.tile_ae_inout


tile_slices_to_image = apc_od.imaging.tile_slices_to_image
tile_slices_to_image_uint8 = apc_od.imaging.tile_slices_to_image_uint8
