#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sklearn.preprocessing as skpre
import sklearn.semi_supervised as sksemi

from lib import *


def preprocess_features(X):
    X = skpre.StandardScaler().fit_transform(X)


def load_data():
    global Xtrain, Ytrain, Xtest, Xvalidate
    Xtrain, Ytrain = load_X('train'), load_Y('train')
    print 'Xtrain, Ytrain:', Xtrain.shape, Ytrain.shape

    Xtest, Xvalidate = load_X('test'), load_X('validate')
    print 'Xtest, Xvalidate:', Xtest.shape, Xvalidate.shape

    preprocess_features(Xtrain)
    preprocess_features(Xtest)
    preprocess_features(Xvalidate)

load_data()
clf = sksemi.LabelPropagation()
X_, Y_ = data_subset(Xtrain, Ytrain, 0.5)
print 'X_, Y_:', X_.shape, Y_.shape
clf.fit(X_, Y_.ravel())

Ypred = clf.predict_proba(Xvalidate)

write_Y('validate', Ypred)
