import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as mplt
import csv
import sklearn.cross_validation as skcv
import sklearn.linear_model as sklin
import sklearn.grid_search as skgs

import datetime

def load_data(fname):
	data = pd.read_csv('data/%s.csv' % fname,
                       index_col=False,
                       header=None,
                       names=['date', 'A', 'B', 'C', 'D', 'E', 'F'])
	data['date'] = pd.to_datetime(data['date'])
    
	data['month'] = data['date'].apply(lambda x: x.month)

	data['weekday'] = data['date'].apply(lambda x: x.isoweekday())

	data['minute'] = data['date'].apply(lambda x: x.minute)
    #~ # separate in two bins, weekend - working days
	for i in range(0, data['weekday'].size):
		if data['weekday'][i]< 5:
			data['weekday'][i]  = 0
		else:
			data['weekday'][i]= 1
    
		
		
		
	
	data['hour'] = data['date'].apply(lambda x: x.hour)
	del data['date']
	return data

		
X = load_data('train')
Y = np.genfromtxt('data/train_y.csv', delimiter=',')

X_m = X['month']
X_m = np.atleast_2d(X_m).T
Y = np.atleast_2d(Y).T
print Y.shape
print X_m.shape
perm = np.argsort(X_m, axis=0)

X_sorted =X_m[perm]
print X_sorted

mplt.title('month')
mplt.plot ( np.squeeze(Y[perm]),'bo')
mplt.show()