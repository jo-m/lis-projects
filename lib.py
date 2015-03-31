#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import unicode_literals
import numpy as np
import h5py


def load_data(fname):
    f = h5py.File('data/%s.h5' % fname, 'r')
    print f.name
    print f.keys()
    if 'label' in f.keys():
        return f['data'], f['label']
    else:
        return f['data'], None


def write_Y(fname, Y):
    if Y.shape[1] != 1:
        raise 'Y has invalid shape!'
    np.savetxt('results/%s_y_pred.txt' % fname, Y, fmt='%d', delimiter=',')


def score(Ytruth, Ypred):
    if Ytruth.shape[1] != 1:
        raise 'Ytruth has invalid shape!'
    if Ypred.shape[1] != 1:
        raise 'Ypred has invalid shape!'

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
