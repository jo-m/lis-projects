#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import unicode_literals
import numpy as np

import h5py


def to_ndarray(h5_arr):
    arr = np.zeros(h5_arr.shape, dtype='float32')
    h5_arr.read_direct(arr)
    return arr


def load_data(fname):
    f = h5py.File('data/%s.h5' % fname, 'r')
    if 'label' in f.keys():
        return to_ndarray(f['data']), to_ndarray(f['label'])
    else:
        return to_ndarray(f['data']), None


def write_Y(fname, Y):
    Y = Y.ravel()
    if Y.ndim != 1:
        raise Exception('Y has invalid shape!')
    if Y.dtype != 'int32':
        raise Exception('Y has invalid dtype!')
    np.savetxt('results/%s_y_pred.txt' % fname,
               Y, fmt=str('%d'), delimiter=',')


def score(Ytruth, Ypred):
    Ytruth = Ytruth.ravel()
    Ypred = Ypred.ravel()
    if Ytruth.ndim != 1:
        raise Exception('Ytruth has invalid shape!')
    if Ypred.ndim != 1:
        raise Exception('Ypred has invalid shape!')

    sum = (Ytruth != Ypred).astype(float).sum().sum()
    return sum / np.product(Ytruth.shape)


def grade(score):
    BE = 0.2778
    BH = 0.1791
    if score > BE:
        return 0
    elif score <= BH:
        return 100
    else:
        return (1 - (score - BH) / (BE - BH)) * 50 + 50
