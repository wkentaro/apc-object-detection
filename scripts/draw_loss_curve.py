#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from apc_od import draw_loss_curve

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--id-num', default=1, type=int,
                        help='number of loss id')
    parser.add_argument('--log-file', type=str, required=True,
                        help='log filename')
    parser.add_argument('--out-file', type=str, required=True,
                        help='output filename')
    parser.add_argument('--no-acc', action='store_true', help='no accuracy')
    args = parser.parse_args()

    n_id = args.id_num
    log_file = args.log_file
    out_file = args.out_file
    no_acc = args.no_acc

    for i in xrange(n_id):
        draw_loss_curve(i, log_file, out_file, no_acc)