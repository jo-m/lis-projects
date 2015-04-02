#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sklearn.cross_validation as skcv
import sklearn.metrics as skmet
import sklearn.preprocessing as skpre

import mlp

from lib import *



def preprocess_features(X):
    pass


class OurClassifier(object):
    n_est = None
    threshold = None

    def __init__(self, n_est=10, threshold='1*mean'):
        self.n_est = n_est
        self.threshold = threshold
        self.clf = mlp.MLPClassifier()

    def fit(self, X, Y):
        X = skpre.StandardScaler().fit_transform(X)
        self.clf.fit(X, Y)
        return self

    def predict(self, X):
        X = skpre.StandardScaler().fit_transform(X)
        return self.clf.predict(X)

    def get_params(self, *x, **xx):
        return {}


def testset_validate(clf):
    Xtrain, Xtest, Ytrain, Ytest = \
        skcv.train_test_split(X, Y, train_size=0.8)
    clf.fit(Xtrain, Ytrain)
    Ypred = clf.predict(Xtest)
    sc = score(Ytest, Ypred)
    print 'Testset score = %.4f Grade = %d%%' % (sc, grade(sc))


def cross_validate(clf):
    scores = skcv.cross_val_score(clf, X, Y, cv=8, n_jobs=1,
                                  scoring=skmet.make_scorer(score))
    print 'C-V score = %.4f ± %.4f Grade = %d%%' % \
        (np.mean(scores), np.std(scores), grade(np.mean(scores)))


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
cross_validate(clf)
