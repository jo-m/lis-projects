#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from sklearn.externals import joblib
import numpy as np
import os
import pandas as pd
import sklearn.cross_validation as skcv
import time

pickle_fname = 'data/clf/%03d.dump'


def try_load_clf(X):
    n = X.shape[0]
    fname = pickle_fname % n
    if os.path.isfile(fname):
        print 'load clf from "%s"' % fname
        return joblib.load(fname)


def save_clf(clf, X, overwrite=False):
    n = X.shape[0]
    fname = pickle_fname % n
    if os.path.isfile(fname) and not overwrite:
        print '"%s" already exists, not overwriting' % fname
        return
    print 'save clf to "%s"' % fname
    joblib.dump(clf, fname)


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)


def data_subset(X, Y, factor=0.5):
    if isinstance(factor, float) and factor == 1.0:
        return X, Y
    subX, _, subY, _ = \
        skcv.train_test_split(X, Y, train_size=factor)
    return subX, subY


def load_X(fname):
    data = pd.read_csv('data/%s.csv' % fname,
                       index_col=False,
                       dtype=np.float64,
                       header=None)
    return data.as_matrix()


def load_Y(fname):
    return pd.read_csv('data/%s_y.csv' % fname,
                       index_col=False,
                       header=None).as_matrix()


# def write_Y(fname, Y):
#     if Y.shape[1] != 2:
#         raise 'Y has invalid shape!'
#     np.savetxt('results/%s_y_pred.txt' % fname, Y, fmt='%d', delimiter=',')


# def score(Ytruth, Ypred):
#     if Ytruth.shape[1] != 2:
#         raise 'Ytruth has invalid shape!'
#     if Ypred.shape[1] != 2:
#         raise 'Ypred has invalid shape!'

#     sum = (Ytruth != Ypred).astype(float).sum().sum()
#     return sum / np.product(Ytruth.shape)


def grade(score):
    BE = 1.1478864255656418
    BH = 0.41876196535569227
    if score > BE:
        return 0
    elif score <= BH:
        return 100
    else:
        return (1 - (score - BH) / (BE - BH)) * 50 + 50
