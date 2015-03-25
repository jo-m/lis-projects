#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from sklearn import ensemble

from lib import *


X, Y = load_X('train'), load_Y('train')
Xtrain, Xtest, Ytrain, Ytest = train_test_split_pd(X, Y, train_size=.8)
print 'Training data: ', Xtrain.shape, Ytrain.shape
print 'Test data: ', Xtest.shape, Ytest.shape

clf = ensemble.RandomForestClassifier(n_estimators=50)
clf.fit(Xtrain, Ytrain)

# calculate score with test set
Ypred = clf.predict(Xtest)
print 'Score:', score(Ytest, Ypred), 'Grade: %d%%' % grade(score(Ytest, Ypred))


def predict_validation_set(clf):
    # load, predict, write validation set
    # train again with whole X, Y
    clf.fit(X, Y)
    Xvalidate = load_X('validate')
    Yvalidate = clf.predict(Xvalidate)
    write_Y('validate', Yvalidate)

predict_validation_set(clf)
