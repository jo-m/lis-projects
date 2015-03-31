#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import Image

from lib import *


def print_row_stats(X, name, nrows=30, Y=None):
    def getlabel(i):
        if Y is None:
            return 'none'
        return '%02d' % Y[i]

    for i in range(nrows):
        print 'X[%d]: mean=%f, std=%f, min=%f, max=%f' % \
            (i, X[i].mean(), X[i].std(), X[i].min(), X[i].max())
        img = Image.fromarray(np.reshape(X[i], (2 ** 5, 2**6)))
        img = img.convert('RGB')
        img.save("img/X_%s_%03d_l%s.png" % (name, i, getlabel(i)))


X, Y = load_data('train')
print '--- train data (X and labels) ---'
print 'data:', X.shape
print 'label:', Y.shape
print_row_stats(X, 'train', Y=Y)

X, _ = load_data('test')
print '--- test data (only X) ---'
print 'data:', X.shape
print_row_stats(X, 'test')

X, _ = load_data('validate')
print '--- validation data (only X) ---'
print 'data:', X.shape
print_row_stats(X, 'validate')
