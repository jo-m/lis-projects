#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sklearn.cross_validation as skcv
import sklearn.ensemble as skens
import sklearn.preprocessing as skpre

from lib import *


def preprocess_features(X):
    pass


class OurClassifier(object):
    n_est = None
    threshold = None

    def __init__(self, n_est=10, threshold='1*mean'):
        self.n_est = n_est
        self.threshold = threshold
        self.clf = skens.RandomForestClassifier(n_estimators=self.n_est,
                                                n_jobs=-1)
        self.trsf = skens.RandomForestClassifier(n_estimators=self.n_est,
                                                 n_jobs=-1)

    def fit(self, X, Y):
        X = skpre.StandardScaler().fit_transform(X)

        self.trsf.fit(X, Y)
        Xred = self.trsf.transform(X, threshold=self.threshold)

        print 'X -> Xred: %d to %d' % \
            (X.shape[1], Xred.shape[1])

        self.clf.fit(Xred, Y)
        return self

    def predict(self, X):
        X = skpre.StandardScaler().fit_transform(X)
        Xred = self.trsf.transform(X, threshold=self.threshold)
        return self.clf.predict(Xred)

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

clf = OurClassifier(threshold='2*mean')
testset_validate(clf)
predict_validation_set(clf)
