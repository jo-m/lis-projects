#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
import dateutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib

"""
http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#example-linear-model-plot-ols-py
http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols_ridge_variance.html#example-linear-model-plot-ols-ridge-variance-py
http://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html
http://stats.stackexchange.com/questions/58739/polynomial-regression-using-scikit-learn
http://www.datarobot.com/blog/regularized-linear-regression-with-scikit-learn/
"""


def load_data(fname):
    data = pd.read_csv('data/%s.csv' % fname,
                       index_col=False,
                       header=None,
                       names=['date', 'A', 'B', 'C', 'D', 'E', 'F'])
    data['date'] = data['date'].apply(dateutil.parser.parse)
    data['date'] = data['date'].apply(matplotlib.dates.date2num)
    return data

train = load_data('train')
train_y = pd.read_csv('data/train_y.csv',
                      index_col=False,
                      header=None,
                      names=['y'])

fig, ax1 = plt.subplots()
ax1.plot_date(train['date'], train_y, 'b,')
ax1.set_xlabel('time')
ax1.set_ylabel('train_y', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
ax2.plot_date(train['date'], train['A'], 'r,')
ax2.set_ylabel('F', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')

ax2 = ax1.twinx()
ax2.plot_date(train['date'], train['F'], 'g,')
ax2.set_ylabel('C', color='g')
for tl in ax2.get_yticklabels():
    tl.set_color('g')


plt.show()

# plt.savefig('graph.pdf', format='pdf')
plt.savefig('graph.png')
