# flake8: NOQA
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tempfile

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


def test_draw_loss_curve():
    # log of supervised training
    supervised_log = '''\
2015-12-01 21:33:06,665 [INFO] epoch:00; train mean loss=inf; accuracy=0.793402775527;
2015-12-01 21:33:08,554 [INFO] epoch:00; test mean loss=0.0223661520096; accuracy=0.99583333234;
2015-12-01 21:33:36,166 [INFO] epoch:01; train mean loss=0.267008397272; accuracy=0.943055551189;
2015-12-01 21:33:38,054 [INFO] epoch:01; test mean loss=0.0319371443122; accuracy=0.99027777546;
2015-12-01 21:34:05,669 [INFO] epoch:02; train mean loss=0.162450927842; accuracy=0.967013884543;
2015-12-01 21:34:07,559 [INFO] epoch:02; test mean loss=0.00178737615885; accuracy=0.99861111078;
2015-12-01 21:34:35,193 [INFO] epoch:03; train mean loss=0.25237462795; accuracy=0.968055551458;
2015-12-01 21:34:37,084 [INFO] epoch:03; test mean loss=2.72755467041e-05; accuracy=1.0;
2015-12-01 21:35:04,718 [INFO] epoch:04; train mean loss=0.270737466685; accuracy=0.967361107469;
2015-12-01 21:35:06,612 [INFO] epoch:04; test mean loss=0.0172848692339; accuracy=0.99444444312;
2015-12-01 21:35:37,238 [INFO] epoch:05; train mean loss=0.138665390847; accuracy=0.978124997682;
2015-12-01 21:35:39,126 [INFO] epoch:05; test mean loss=1.31297177821e-07; accuracy=1.0;
'''
    log_file = tempfile.mktemp()
    with open(log_file, 'w') as f:
        f.write(supervised_log)
    apc_od.draw_loss_curve(log_file, tempfile.mktemp(), no_acc=False)
    # log of unsupervised training
    unsupervised_log = '''\
2015-11-24 23:57:03,540 [INFO] epoch:00; train mean loss=0.00389542509668;
2015-11-24 23:57:07,646 [INFO] epoch:00; test mean loss=0.000338642162872;
2015-11-24 23:57:25,875 [INFO] epoch:01; train mean loss=0.000212499197854;
2015-11-24 23:57:30,128 [INFO] epoch:01; test mean loss=0.000117461651098;
2015-11-24 23:57:48,356 [INFO] epoch:02; train mean loss=8.24619589366e-05;
2015-11-24 23:57:52,628 [INFO] epoch:02; test mean loss=5.45797932015e-05;
2015-11-24 23:58:10,803 [INFO] epoch:03; train mean loss=4.39608514929e-05;
2015-11-24 23:58:15,035 [INFO] epoch:03; test mean loss=3.45692531482e-05;
2015-11-24 23:58:33,109 [INFO] epoch:04; train mean loss=2.96583379774e-05;
2015-11-24 23:58:37,317 [INFO] epoch:04; test mean loss=2.53010121924e-05;
2015-11-24 23:58:55,482 [INFO] epoch:05; train mean loss=2.26222944017e-05;
2015-11-24 23:58:59,687 [INFO] epoch:05; test mean loss=2.04618284746e-05;
'''
    log_file = tempfile.mktemp()
    with open(log_file, 'w') as f:
        f.write(unsupervised_log)
    apc_od.draw_loss_curve(log_file, tempfile.mktemp(), no_acc=True)

