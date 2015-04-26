#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sklearn.cross_validation as skcv
import sklearn.preprocessing as skpre


from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from lib import *


X, Y = load_data('train')
X = skpre.StandardScaler().fit_transform(X)

Xtrain, Xtest, Ytrain, Ytest = \
    skcv.train_test_split(X, Y, train_size=0.8)

lb = skpre.LabelBinarizer()
lb.fit(Ytrain)

num_classes = len(lb.classes_)
num_features = Xtrain.shape[1]

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('output', DenseLayer)]


clf = NeuralNet(layers=layers0,

                dense0_num_units=200,
                dropout0_p=0.5,
                dense1_num_units=200,

                input_shape=(None, num_features),
                output_num_units=num_classes,
                output_nonlinearity=softmax,

                update=nesterov_momentum,
                update_learning_rate=0.01,
                update_momentum=0.9,

                eval_size=0.2,
                verbose=1,
                max_epochs=200,
                regression=False)


with Timer('testset'):
    clf.fit(Xtrain, Ytrain)
    Ypred = clf.predict(Xtest)
    sc = score(Ytest, Ypred)
    print 'Testset score = %.4f Grade = %d%%' % (sc, grade(sc))

quit()

clf.fit(X, Y)
Xvalidate, _ = load_data('validate')
Xvalidate = skpre.StandardScaler().fit_transform(Xvalidate)
Yvalidate = clf.predict(Xvalidate)
write_Y('validate', Yvalidate)
