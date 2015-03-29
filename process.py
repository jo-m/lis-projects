#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pandas as pd
import sklearn.cross_validation as skcv
import sklearn.ensemble as skens
import sklearn.metrics as skmet
import sklearn.preprocessing as skpre

from lib import *


def preprocess_features(X):
    del X['B']


class UseY1Classifier(object):
    threshold = None

    def __init__(self, n_est=100, threshold='mean'):
        # we need 2 separate classifiers
        self.clf1 = skens.RandomForestClassifier(n_estimators=n_est)
        self.clf2 = skens.RandomForestClassifier(n_estimators=n_est)

        # random forests to throw out unimportant features
        self.threshold = threshold
        self.trsf1 = skens.RandomForestClassifier(n_estimators=n_est)
        self.trsf2 = skens.RandomForestClassifier(n_estimators=n_est)

    def _binarize(self, y):
        y = np.atleast_2d(y).T
        enc = skpre.OneHotEncoder(sparse=False)
        enc.fit(y)
        return enc.transform(y)

    def fit(self, X, Y):
        if isinstance(Y, pd.DataFrame):
            # make a numpy ndarray out of pandas dataframe
            Y = Y.as_matrix()
            X = X.as_matrix()
        # append y1 to X
        X_y1 = np.concatenate([X, self._binarize(Y[:, 0])], axis=1)

        # transform
        # reduce number of features
        self.trsf1.fit(X, Y[:, 0])
        self.trsf2.fit(X_y1, Y[:, 1])

        old = X.shape[1], X_y1.shape[1]
        X_for_y1 = self.trsf1.transform(X, threshold=self.threshold)
        X_for_y2 = self.trsf2.transform(X_y1, threshold=self.threshold)
        print 'y1: %d to %d, y2: %d to %d' % \
            (old[0], X_for_y1.shape[1], old[1], X_for_y2.shape[1])

        # normalize
        X_for_y1 = skpre.StandardScaler().fit_transform(X_for_y1)
        X_for_y2 = skpre.StandardScaler().fit_transform(X_for_y2)

        # fit X vs y1
        self.clf1.fit(X_for_y1, Y[:, 0])
        # fit X + y1 vs y2
        self.clf2.fit(X_for_y2, Y[:, 1])
        return self

    def predict(self, X):
        X_for_y1 = self.trsf1.transform(X, threshold=self.threshold)
        X_for_y1 = skpre.StandardScaler().fit_transform(X_for_y1)

        # pred y1 from X
        y1 = self.clf1.predict(X_for_y1)
        X_y1 = np.concatenate([X, self._binarize(y1)], axis=1)
        X_for_y2 = self.trsf2.transform(X_y1, threshold=self.threshold)
        X_for_y2 = skpre.StandardScaler().fit_transform(X_for_y2)
        # pred y2 from X + y1
        y2 = self.clf2.predict(X_for_y2)
        return np.vstack([y1, y2]).T

    def get_params(self, *x, **xx):
        return {}


def testset_validate(clf):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split_pd(X, Y, train_size=.8)
    clf.fit(Xtrain, Ytrain)
    Ypred = clf.predict(Xtest)
    sc = score(Ytest, Ypred)
    print 'Testset score = %.4f Grade = %d%%' % (sc, grade(sc))


def cross_validate(clf):
    scores = skcv.cross_val_score(clf, X, Y, cv=8, n_jobs=-1,
                                  scoring=skmet.make_scorer(score))
    print 'C-V score = %.4f ± %.4f Grade = %d%%' % \
        (np.mean(scores), np.std(scores), grade(np.mean(scores)))


def predict_validation_set(clf):
    clf.fit(X, Y)
    Xvalidate = load_X('validate')
    preprocess_features(Xvalidate)
    Yvalidate = clf.predict(Xvalidate)
    write_Y('validate', Yvalidate)

X, Y = load_X('train'), load_Y('train')
preprocess_features(X)

clf = UseY1Classifier(50)
testset_validate(clf)
cross_validate(clf)
predict_validation_set(clf)
