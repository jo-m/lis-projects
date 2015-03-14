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


from sklearn.linear_model import LinearRegression
import sklearn.cross_validation as skcv



def load_data(fname):
    data = pd.read_csv('data/%s.csv' % fname,
                       index_col=False,
                       header=None,
                       names=['date', 'A', 'B', 'C', 'D', 'E', 'F'])
                     
    data['date'] = data['date'].apply(dateutil.parser.parse)    
    data['weekday'] = data['date'].apply(dt.datetime.weekday)
    data['weekend'] = data['date'].apply(dt.datetime.weekday) >4
    data['month'] = data['date'].apply(lambda x:x.month)
    data['hour'] = data['date'].apply(lambda x:x.hour)
    data['year'] = data['date'].apply(lambda x:x.year)
    data['dayofyear'] = data['date'].apply(lambda x:x.dayofyear)
       
    
    del data['date']
    #del data['dayofyear']
    del data['month']
    #del data['weekday']
    #del data['year']
    #del data['hour']
    #del data['weekend']
    #del data['A']
    #del data['B']
    #del data['C']
    #del data['D']
    #del data['E']
    #del data['F']
    
    
    return data


def apply_polynominals(X, column, p=30):
    for i in range(2, p + 1):
        X['%s^%d' % (column, i)] = np.power(X[column], i)
  
    
def transformFeatures(X):
    
    apply_polynominals(X, 'hour', 10)
    
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
    regressor = LinearRegression()  
    regressor.fit(Xtrain,np.log(Ytrain))
    
    Ypred = np.array(regressor.predict(Xtest),dtype=float) 
    
    print logscore( Ytest, np.exp(Ypred ) )
        
    validate = load_data('validate')
    validate = transformFeatures(validate)    
    np.savetxt('results/validate.txt', np.exp(np.array( regressor.predict(validate), dtype=float)))  
 

X = load_data('train')
X = transformFeatures(X)
Y = pd.read_csv('data/train_y.csv',
                index_col=False,
                header=None,
                names=['y'])


Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=.75)

linregTrans()


