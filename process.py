#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sklearn.preprocessing as skpre
import sklearn.semi_supervised as sksemi

from lib import *


def preprocess_features(X):
    X = skpre.StandardScaler().fit_transform(X)


def load_data():
    global X, Y, X_, Y_, Xtrain, Ytrain, Xtest, Xvalidate
    Xtrain, Ytrain = load_X('train'), load_Y('train')
    print 'Xtrain, Ytrain:', Xtrain.shape, Ytrain.shape

    Xtest, Xvalidate = load_X('test'), load_X('validate')
    print 'Xtest, Xvalidate:', Xtest.shape, Xvalidate.shape
    X = np.concatenate((Xtrain, Xtest, Xvalidate))

    diff = X.shape[0] - Xtrain.shape[0]
    Y = np.concatenate((Ytrain, np.atleast_2d(np.repeat(-1, diff)).T))

    preprocess_features(X)

    print 'X, Y:', X.shape, Y.shape
    X_, Y_ = data_subset(X, Y, 1.0)
    print 'X_, Y_:', X_.shape, Y_.shape

load_data()
clf = sksemi.LabelPropagation()
clf.fit(Xtrain, Ytrain)

Ypred = clf.predict_proba(Xvalidate)
print Ypred.shape

write_Y('validate', Ypred)
