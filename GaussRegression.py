# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 16:29:13 2015

@author: Andreas
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 16:23:40 2015

@author: Andreas
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import csv
import matplotlib 
import datetime as dt
import dateutil

import sklearn.cross_validation as skcv



def load_data(inpath):
    X=[]
    with open('./data/%s.csv' % inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            t=dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
                       
            X.append( t.hour )
        X=np.array(X,dtype=float)
        return np.atleast_2d(X)


#def load_data(fname):
#    data = pd.read_csv('data/%s.csv' % fname,
#                       index_col=False,
#                       header=None,
#                       names=['date', 'A', 'B', 'C', 'D', 'E', 'F'])
#                     
#    data['date'] = data['date'].apply(dateutil.parser.parse)    
#    data['weekday0'] = data['date'].apply(dt.datetime.weekday)
#    data['month'] = data['date'].apply(lambda x:x.month)
#    data['hour0'] = data['date'].apply(lambda x:x.hour)
#    #data['year0'] = data['date'].apply(lambda x:x.year)
#    
#    del data['date']
#    del data['month']
#    del data['weekday0']
#    del data['A']
#    del data['B']
#    del data['C']
#    del data['D']
#    del data['E']
#    del data['F']
#        
#
#    return data

    

X = np.transpose(load_data('train'))

Y = pd.read_csv('data/train_y.csv',
                index_col=False,
                header=None,
                names=['y'])


X = np.atleast_2d(X)
Y = np.atleast_2d(Y)


Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=.1)



def logscore(gtruth, pred):
    pred = np.clip(pred, 0, np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + pred)
    return np.sqrt(np.mean(np.square(logdif)))
            
            

from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl

gp = GaussianProcess( corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)

print Xtrain.shape
print Ytrain.shape
gp.fit(Xtrain, Ytrain)

Ypred = gp.predict(Xtrain) #todo what is mesh?

print 'score =' + logscore(Ypred,Ytest)











