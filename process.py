@@ -5,9 +5,12 @@ import pandas as pd
import numpy as np
import dateutil
import matplotlib
# plot to file, for use in VM
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import sklearn.cross_validation as skcv


"""
http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#example-linear-model-plot-ols-py
http://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html
http://stats.stackexchange.com/questions/58739/polynomial-regression-using-scikit-learn
http://www.datarobot.com/blog/regularized-linear-regression-with-scikit-learn/

- first step: simple regressor, like in class example
- constrain regressor to positive numbers
- categorical values: what to do with them? There is no ordering!
  (Add dimensions for them?)
- use weekdays
- feature engineering
"""


def load_data(fname):
    data = pd.read_csv('data/%s.csv' % fname,
                       index_col=False,
                       header=None,
                       names=['date', 'A', 'B', 'C', 'D', 'E', 'F'])

    # date transformations
    data['date'] = pd.to_datetime(data['date'])
    data['dayofyear'] = data['date'].apply(lambda x: x.dayofyear)
    data['weekday'] = data['date'].apply(lambda x: x.isoweekday())
    data['hour'] = data['date'].apply(lambda x: x.hour)
    data['min'] = data['date'].apply(lambda x: x.minute)
    data['date'] = data['date'].apply(matplotlib.dates.date2num)
    del data['date']

    return data

train = load_data('train')
train_y = pd.read_csv('data/train_y.csv',
                      index_col=False,
                      header=None,
                      names=['y'])
X = load_data('train')
Y = pd.read_csv('data/train_y.csv',
                index_col=False,
                header=None,
                names=['y'])

fig, ax1 = plt.subplots()
ax1.plot_date(train['date'], train_y, 'b,')
ax1.set_xlabel('time')
ax1.set_ylabel('train_y', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')
Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=.75)

ax2 = ax1.twinx()
ax2.plot_date(train['date'], train['A'], 'r,')
ax2.set_ylabel('F', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.plot(Xtest[:, 0], Ytest, 'b,')
plt.show()
plt.savefig('graph.png')

ax2 = ax1.twinx()
ax2.plot_date(train['date'], train['F'], 'g,')
ax2.set_ylabel('C', color='g')
for tl in ax2.get_yticklabels():
    tl.set_color('g')
import sklearn.linear_model as sklin
regressor = sklin.Ridge()
regressor.fit(Xtrain, Ytrain)
print regressor.coef_


plt.show()
def logscore(gtruth, pred):
    pred = np.clip(pred, 0, np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + pred)
    return np.sqrt(np.mean(np.square(logdif)))

# plt.savefig('graph.pdf', format='pdf')
plt.savefig('graph.png')

Ypred = regressor.predict(Xtest)
print 'score =', logscore(Ytest, Ypred)


# use cross validation, eg 5 sets, calc mean and variance
import sklearn.metrics as skmet
scorefun = skmet.make_scorer(logscore)
scores = skcv.cross_val_score(regressor, X, Y, scoring=scorefun, cv=5)
print scores
print np.mean(scores), np.var(scores)

# print validation data set
validate = load_data('validate')
np.savetxt('results/validate.txt', regressor.predict(validate))

# scikit parameters auto
import sklearn.grid_search as skgs
regressor_ridge = sklin.Ridge()
param_grid = dict(alpha=[1, 10, 50, 100])
# bcs gridsearch tries to maximize but we want to minimize
neg_scorefun = skmet.make_scorer(lambda x, y: -logscore(x, y))
grid_search = skgs.GridSearchCV(regressor_ridge,
                                param_grid,
                                scoring=neg_scorefun,
                                cv=5)
grid_search.fit(X, Y)
print grid_search.best_estimator_
