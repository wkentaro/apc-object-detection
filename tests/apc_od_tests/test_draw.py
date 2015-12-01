#!/usr/bin/env python
# -*- coding: utf-8 -*-

import apc_od


def test_tile_ae_inout():
    im = apc_od.data.doll()
    x = apc_od.im_to_blob(im)
    x_hat = x.copy()
    apc_od.tile_ae_inout([x], [x_hat], '/tmp/tile_ae_inout.jpg')


def test_tile_ae_encoded():
    im = apc_od.data.doll()
    x = apc_od.im_to_blob(im)
    apc_od.tile_ae_encoded([x, x], '/tmp/tile_ae_encoded.jpg')