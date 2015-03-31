#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sklearn.cross_validation as skcv
import sklearn.ensemble as skens
import sklearn.preprocessing as skpre

from lib import *


def preprocess_features(X):
    pass


class OurClassifier(object):
    n_est = 200

    def __init__(self):
        self.clf = skens.RandomForestClassifier(n_estimators=self.n_est,
                                                n_jobs=-1)

    def fit(self, X, Y):
        X = skpre.StandardScaler().fit_transform(X)
        self.clf.fit(X, Y.ravel())
        return self

    def predict(self, X):
        X = skpre.StandardScaler().fit_transform(X)
        return self.clf.predict(X)

    def get_params(self, *x, **xx):
        return {}


def testset_validate(clf):
    Xtrain, Xtest, Ytrain, Ytest = \
        skcv.train_test_split(X, Y, train_size=0.01)
    clf.fit(Xtrain, Ytrain)
    Ypred = clf.predict(Xtest)
    sc = score(Ytest, Ypred)
    print 'Testset score = %.4f Grade = %d%%' % (sc, grade(sc))


def predict_validation_set(clf):
    clf.fit(X, Y)
    Xvalidate, _ = load_data('validate')
    preprocess_features(Xvalidate)
    Yvalidate = clf.predict(Xvalidate)
    write_Y('validate', Yvalidate)

X, Y = load_data('train')
preprocess_features(X)

clf = OurClassifier()
testset_validate(clf)
predict_validation_set(clf)
