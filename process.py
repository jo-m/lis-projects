#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lib import *


X = load_X('train')
Y = load_Y('train')

print X.head()
print Y.head()

print X.shape
print Y.shape

Xtrain, Xtest, Ytrain, Ytest = train_test_split_pd(X, Y, train_size=.8)

print Xtrain.shape
print Ytrain.shape
