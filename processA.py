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



#def load_data(inpath):
#    X=[]
#    with open('./data/%s.csv' % inpath, 'r') as fin:
#        reader = csv.reader(fin, delimiter=',')
#        for row in reader:
#            t=datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
#            XFirstOrder = np.hstack((t.weekday(), t.month, t.hour ,row[1:6],1, row[1]*row[1] ) )
#            
#            X.append( XFirstOrder )
#        X=np.array(X,dtype=float)
#        return np.atleast_2d(X)


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



def makePoly(X):
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
 

def ridgereg():
    
    
    import sklearn.linear_model as sklin
    
    #Ridge Regressor
    import sklearn.metrics as skmet
    import sklearn.grid_search as skgs

    regressor_ridge = sklin.Ridge() #neuen Ridge regressor anlegen
    
    param_grid = dict(alpha=np.linspace(0,1000,num=5)) #diese Parameter werden darauf untersucht wie gut sie sind
    neg_scorefun = skmet.make_scorer(lambda x, y: -logscore(x, y)) #anstatt der orginalfunktion verwenden wir unsere funktion, aber negativ, damit sie minimiert nicht maximiert wrid
    
    grid_search = skgs.GridSearchCV( regressor_ridge, param_grid, scoring=neg_scorefun, cv=5)
    grid_search.fit(X,Y) #Ohne Aufteilung in Sets, da das von grid search übernommen wird? -> Ja von cv
    
    print grid_search.best_estimator_
        
    Ypred = grid_search.predict(Xtest)
        
    print logscore( Ytest, Ypred) 


def linreg():
    

    import sklearn.linear_model as sklin
    
    #Ridge Regressor
    import sklearn.metrics as skmet
    import sklearn.grid_search as skgs    
    
    regressor = sklin.LinearRegression() #neuen Ridge regressor anlegen
    regressor.fit(Xtrain,Ytrain) #Ohne Aufteilung in Sets, da das von grid search übernommen wird? -> Ja von cv
    
    #scorefun = skmet.make_scorer(logscore)
    #scores = skcv.cross_val_score(regressor, X, Y, scoring=scorefun, cv=5)   
    
    #print scores
    
    Ypred = regressor.predict(Xtest)

    print logscore( Ytest, Ypred) 


def polyreg():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.decomposition import PCA
    

    model = make_pipeline( PolynomialFeatures(30), LinearRegression() )
    
    model.fit(Xtrain,Ytrain) #Ohne Aufteilung in Sets, da das von grid search übernommen wird? -> Ja von cv
    
    #scorefun = skmet.make_scorer(logscore)
    #scores = skcv.cross_val_score(regressor, X, Y, scoring=scorefun, cv=5)   
    
    #print scores
    
    Ypred = model.predict(Xtest)

    print logscore( Ytest, Ypred) 


def svmReg():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    
    estimators = [('reduce_dim', PCA()), ('svm', SVC())]
    clf = Pipeline(estimators)
    clf.fit(Xtrain, Ytrain)
    
    Ypred = clf.predict(Xtest)

    print logscore( Ytest, Ypred) 

def linWithDimReg():
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.decomposition import PCA
    
    estimators = [('reduce_dim', PCA()), ('poly', PolynomialFeatures(10)),  ('ridge', Lasso())]
    
    clf = Pipeline(estimators)
    clf.fit(Xtrain, Ytrain)
    
    Ypred = clf.predict(Xtest)

    print logscore( Ytest, Ypred) 


def linregTrans():
    

    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
       
    
    estimators = [('poly', PolynomialFeatures(10)),  ('linear', LinearRegression())]
    
    regressor = Pipeline(estimators)
    
    regressor.fit(Xtrain,np.log(Ytrain)) #Ohne Aufteilung in Sets, da das von grid search übernommen wird? -> Ja von cv
    
    #scorefun = skmet.make_scorer(logscore)
    #scores = skcv.cross_val_score(regressor, X, Y, scoring=scorefun, cv=5)   
    
    #print scores
    
    Ypred = np.array(regressor.predict(Xtest),dtype=float) 
    
    print logscore( Ytest, np.exp(Ypred ) )
    
    
    validate = load_data('validate')
    np.savetxt('results/validate.txt', np.exp( np.array( regressor.predict(validate), dtype=float) ) )
    
    


X = load_data('train')
#X=makePoly(X)
Y = pd.read_csv('data/train_y.csv',
                index_col=False,
                header=None,
                names=['y'])


Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=.75)

linregTrans()

Y=np.sort(Y, axis=0)

plt.plot(np.log(Y),'.')
plt.show()

            

#import sklearn.linear_model as sklin
#regressor = sklin.Ridge()
#regressor.fit(Xtrain, Ytrain)
#print regressor.coef_      
#
#  
#def logscore(gtruth, pred):
#    pred = np.clip(pred, 0, np.inf)
#    logdif = np.log(1 + gtruth) - np.log(1 + pred)
#    return np.sqrt(np.mean(np.square(logdif)))
#
#
#Ypred = regressor.predict(Xtest)
#print 'score =', logscore(Ytest, Ypred)
#
#
## use cross validation, eg 5 sets, calc mean and variance
#import sklearn.metrics as skmet
#scorefun = skmet.make_scorer(logscore)
#scores = skcv.cross_val_score(regressor, X, Y, scoring=scorefun, cv=5)
#print scores
#print np.mean(scores), np.var(scores)
#
## print validation data set
#validate = load_data3('validate')
#np.savetxt('results/validate.txt', regressor.predict(validate))
#
## scikit parameters auto
#import sklearn.grid_search as skgs
#regressor_ridge = sklin.Ridge()
#param_grid = dict(alpha=[1, 10, 50, 100])
## bcs gridsearch tries to maximize but we want to minimize
#neg_scorefun = skmet.make_scorer(lambda x, y: -logscore(x, y))
#grid_search = skgs.GridSearchCV(regressor_ridge,
#                                param_grid,
#                                scoring=neg_scorefun,
#                                cv=5)
#grid_search.fit(X, Y)
#print grid_search.best_estimator_
#
#
#
#
