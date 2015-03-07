import numpy as np
import matplotlib.pyplot as mplt
import csv
import sklearn.cross_validation as skcv

import datetime

def read_data (inpath):
	X = []
	Xyear = []
	with open(inpath, 'r') as fin:
		reader = csv.reader(fin, delimiter=',')
		for row in reader:
			t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
			X.append(get_features(t.hour))
			Xyear.append(get_features(t.year))
		return np.atleast_2d(X)

def get_features(h): #can be h^2...e
	r = []
	for p in range (1, degree+1):
		r.append(np.power(h, p)) #polynom
	return r

def logscore (gtruth, pred):
	pred = np.clip(pred, 0, np.inf)
	logdif = np.log(1 + gtruth) - np.log(1+pred)
	return np.sqrt(np.mean(np.square(logdif)))

#~ degree = 30 #polynomial degre
degree = 30#polynomial degre
print ('degree', degree)

X = read_data('data/train.csv')
Y = np.genfromtxt('data/train_y.csv', delimiter=',')

print X.shape
print Y.shape
Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X,Y, train_size=0.75)


import sklearn.linear_model as sklin

regressor = sklin.LinearRegression()
regressor.fit(Xtrain, Ytrain)

Hplot = range (25)
Xplot = np.atleast_2d([get_features(x) for x in Hplot])
Yplot = regressor.predict(Xplot)
mplt.plot(Xtrain[:,0], Ytrain, ('bo'))
mplt.plot(Hplot, Yplot,'r', linewidth=3)
mplt.xlim([-0.5, 23.5])
mplt.ylim([-1, 1000])

Ypred = regressor.predict(Xtest)

#~ mplt.plot(Xtest[:,0], Ypred - Ytest, ('go'))
print ('score = ', logscore(Ytest, Ypred))


import sklearn.metrics as skmet
scorefun = skmet.make_scorer(logscore)
scores = skcv.cross_val_score(regressor, X, Y, scoring=scorefun, cv=5)
print ('C-V score = ', np.mean(scores), '+/-', np.std(scores))


mplt.show()
