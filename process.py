#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sklearn.cross_validation as skcv
import sklearn.metrics as skmet
import sklearn.preprocessing as skpre
import sklearn.grid_search as skgs

import mlp

from lib import *


def preprocess_features(X):
    pass


class OurClassifier(object):
    def __init__(self, *x, **xx):
        self.clf = mlp.MLPClassifier(*x, **xx)

    def fit(self, X, Y):
        X = skpre.StandardScaler().fit_transform(X)
        self.clf.fit(X, Y)
        return self

    def predict(self, X):
        X = skpre.StandardScaler().fit_transform(X)
        return self.clf.predict(X)

    def get_params(self, *x, **xx):
        return self.clf.get_params(*x, **xx)

    def set_params(self, *x, **xx):
        self.clf.set_params(*x, **xx)
        return self

    def __str__(self):
        return str(self.clf)


def testset_validate(clf):
    Xtrain, Xtest, Ytrain, Ytest = \
        skcv.train_test_split(X, Y, train_size=0.01)
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


def grid_search(clf):
    Xtrain, Xtest, Ytrain, Ytest = \
        skcv.train_test_split(X, Y, train_size=0.1)

    param_grid = dict(n_hidden=[50, 100, 200])
    # bcs gridsearch tries to maximize but we want to minimize
    neg_scorefun = skmet.make_scorer(neg_score)
    grid_search = skgs.GridSearchCV(clf,
                                    param_grid,
                                    scoring=neg_scorefun,
                                    cv=5,
                                    n_jobs=1)
    grid_search.fit(Xtrain, Ytrain)
    print grid_search.best_estimator_

X, Y = load_data('train')
preprocess_features(X)

clf = OurClassifier(n_hidden=200)
testset_validate(clf)
predict_validation_set(clf)
cross_validate(clf)
grid_search(clf)
