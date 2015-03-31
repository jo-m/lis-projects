#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sklearn.cross_validation as skcv

from lib import *

X, Y = load_data('train')


print X.shape, Y.shape

Xtrain, Xtest, Ytrain, Ytest = \
    skcv.train_test_split(X, Y, train_size=0.8)

print Xtrain.shape, Ytrain.shape
print Xtest.shape, Ytest.shape
