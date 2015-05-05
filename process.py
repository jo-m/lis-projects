#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sklearn.preprocessing as skpre

from lib import *


def preprocess_features(X):
    X = skpre.StandardScaler().fit_transform(X)

X, Y = load_X('train'), load_Y('train')
preprocess_features(X)

print X.shape, Y.shape
