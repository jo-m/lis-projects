#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sklearn.preprocessing as skpre
import sklearn.mixture as skmx


from lib import *


def preprocess_features(X):
    X = skpre.StandardScaler().fit_transform(X)


def load_data():
    global X, Y, X_, Y_
    Xtrain, Ytrain = load_X('train'), load_Y('train')
    print 'Xtrain, Ytrain:', Xtrain.shape, Ytrain.shape

    Xtest, Xvalidate = load_X('test'), load_X('validate')
    print 'Xtest, Xvalidate:', Xtest.shape, Xvalidate.shape
    X = np.concatenate((Xtrain, Xtest, Xvalidate))

    diff = X.shape[0] - Xtrain.shape[0]
    Y = np.concatenate((Ytrain, np.atleast_2d(np.repeat(-3, diff)).T))

    preprocess_features(X)

    print 'X, Y:', X.shape, Y.shape
    X_, Y_ = data_subset(X, Y, 0.01)
    print 'X_, Y_:', X_.shape, Y_.shape
    return X_, Y_

load_data()
clf = try_load_clf(X_)
if clf is None:
    clf = skmx.GMM(n_components=8)
    clf.fit(X_)
    save_clf(clf, X_)
