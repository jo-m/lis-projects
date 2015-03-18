#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm

from lib import *


X = load_X('train')
Y = load_Y('train')
Xtrain, Xtest, Ytrain, Ytest = train_test_split_pd(X, Y, train_size=.003)
print 'Training data: ', Xtrain.shape, Ytrain.shape
print 'Test data: ', Xtest.shape, Ytest.shape


def train(Xtrain, Ytrain, clf1, clf2):
    clf1.fit(Xtrain, Ytrain['y1'])
    print 'Done training y1'
    clf2.fit(Xtrain, Ytrain['y2'])
    print 'Done training y2'


def pred(X, clf1, clf2):
    y1 = clf1.predict(X)
    y2 = clf2.predict(X)
    return pd.DataFrame(dict(y1=y1, y2=y2))

clf1 = svm.SVC(kernel='linear')
clf2 = svm.SVC(kernel='linear')
train(Xtrain, Ytrain, clf1, clf2)

# validate with test set
Ypred = pred(Xtest, clf1, clf2)
print 'Score:', score(Ytest, Ypred), 'Grade: ', grade(score(Ytest, Ypred))

Xvalidate = load_X('validate')
Yvalidate = pred(Xvalidate, clf1, clf2)
write_Y('validate', Yvalidate)
