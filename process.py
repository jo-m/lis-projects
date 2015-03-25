#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sklearn.ensemble as skens
import sklearn.cross_validation as skcv
import sklearn.metrics as skmet

from lib import *


X, Y = load_X('train'), load_Y('train')


class UseY1Classifier(object):
    clf1 = skens.RandomForestClassifier(n_estimators=50)
    clf2 = skens.RandomForestClassifier(n_estimators=50)

    def fit(self, *x, **xx):
        self.clf1.fit(*x, **xx)
        return self

    def predict(self, *x, **xx):
        return self.clf1.predict(*x, **xx)

    def get_params(self, *x, **xx):
        return {}

clf = UseY1Classifier()


def testset_validate(clf):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split_pd(X, Y, train_size=.8)
    clf.fit(Xtrain, Ytrain)
    Ypred = clf.predict(Xtest)
    sc = score(Ytest, Ypred)
    print 'Testset score = %.4f Grade = %d%%' % (sc, grade(sc))


def cross_validate(clf):
    scores = skcv.cross_val_score(clf, X, Y, cv=5,
                                  scoring=skmet.make_scorer(score))
    print 'C-V score = %.4f ± %.4f Grade = %d%%' % \
        (np.mean(scores), np.std(scores), grade(np.mean(scores)))


def predict_validation_set(clf):
    # load, predict, write validation set
    # train again with whole X, Y
    clf.fit(X, Y)
    Xvalidate = load_X('validate')
    Yvalidate = clf.predict(Xvalidate)
    write_Y('validate', Yvalidate)

testset_validate(clf)
cross_validate(clf)
predict_validation_set(clf)
