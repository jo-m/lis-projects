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
    # normalize
    X = skpre.StandardScaler().fit_transform(X)


class UseY1Classifier(object):
    def __init__(self, n_est=100):
        # we need 2 separate classifiers
        self.clf1 = skens.RandomForestClassifier(n_estimators=n_est)
        self.clf2 = skens.RandomForestClassifier(n_estimators=n_est)

    def _trans_y(self, y):
        """
        Binarize y1 so we can use it for prediciton of y2
        """
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
        X_y1 = np.concatenate([X, self._trans_y(Y[:, 0])], axis=1)
        # fit X vs y1
        self.clf1.fit(X, Y[:, 0])
        # fit X + y1 vs y2
        self.clf2.fit(X_y1, Y[:, 1])
        return self

    def predict(self, X):
        # pred y1 from X
        y1 = self.clf1.predict(X)
        X_y1 = np.concatenate([X, self._trans_y(y1)], axis=1)
        # pred y2 from X + y1
        y2 = self.clf2.predict(X_y1)
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
