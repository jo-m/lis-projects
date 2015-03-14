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
import datetime as dt
import dateutil

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sklearn.cross_validation as skcv



def load_data(fname):
    data = pd.read_csv('data/%s.csv' % fname,
                       index_col=False,
                       header=None,
                       names=['date', 'A', 'B', 'C', 'D', 'E', 'F'])
                     
    data['date'] = data['date'].apply(dateutil.parser.parse)    
    data['weekday0'] = data['date'].apply(dt.datetime.weekday)
    data['weekend'] = data['date'].apply(dt.datetime.weekday) >4
    data['notWeekend'] =  (data['weekend'] *-1)+1
    data['month'] = data['date'].apply(lambda x:x.month)
    data['hour0'] = data['date'].apply(lambda x:x.hour)
    data['year0'] = data['date'].apply(lambda x:x.year)
    
     
    
    del data['date']
    del data['month']
    del data['weekday0']
    del data['year0']
    #del data['hour0']
    #del data['weekend']
    #del data['notWeekend']
    del data['A']
    del data['B']
    del data['C']
    del data['D']
    del data['E']
    del data['F']
    
    
    return data


def transformFeatures(X):
    gradHour=30
    for i in range(1,gradHour):
        X[ 'hour'+str(i)] = X['hour0']*X['hour'+str(i-1)]
    gradWeekday=3
    for i in range(1,gradWeekday):
        X[ 'weekday'+str(i)] = X['weekday0']*X['weekday'+str(i-1)]
    gradYear=3
    for i in range(1,gradYear):
        X[ 'year'+str(i)] = X['year0']*X['year'+str(i-1)]
    
    for col in X.columns:
        std=np.std(X[col])
        mean=np.mean(X[col])
        X[col]=(X[col]-mean)/std
        
    return X

    
def logscore(gtruth, pred):
    gtruth=np.array(gtruth,dtype=float)
    pred=np.array(pred,dtype=float)
    pred = np.clip(pred, 0, np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + pred)
    return np.sqrt(np.mean(np.square(logdif)))
 

def linregTrans():
    estimators = [('poly', PolynomialFeatures(10)),  ('linear', LinearRegression())]    
    regressor = Pipeline(estimators)    
    regressor.fit(Xtrain,np.log(Ytrain))
    
    Ypred = np.array(regressor.predict(Xtest),dtype=float) 
    
    print logscore( Ytest, np.exp(Ypred ) )
    
    validate = load_data('validate')
    np.savetxt('results/validate.txt', np.exp( np.array( regressor.predict(validate), dtype=float) ) )
 

X = load_data('train')
#X=transformFeatures(X)
Y = pd.read_csv('data/train_y.csv',
                index_col=False,
                header=None,
                names=['y'])


Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=.75)

linregTrans()


