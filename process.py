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
clf = sksemi.LabelSpreading()
info(Xtrain, 'Xtrain')
info(Ytrain.ravel(), 'Ytrain.ravel()')
clf.fit(Xtrain, Ytrain.ravel())

info(clf.classes_, 'classes_')
info(clf.label_distributions_, 'label_distributions_')
info(clf.transduction_, 'transduction_')

clf.label_distributions_ = np.nan_to_num(clf.label_distributions_)

info(clf.label_distributions_, 'label_distributions_')

Ypred = clf.predict_proba(Xvalidate)

write_Y('validate_spreading', np.nan_to_num(Ypred))
