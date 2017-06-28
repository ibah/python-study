# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:51:31 2017
@author: a

Study: statsmodels

"""


"""
Getting started
"""

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# R-style formulas together with pandas data frames

# Load data
dat = sm.datasets.get_rdataset("Guerry", "HistData").data
# Fit regression model (using the natural log of one of the regressors)
results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()
# Inspect the results
print(results.summary())

# using numpy arrays instead of formulas

# Generate artificial data (2 regressors + constant)
nobs = 100
X = np.random.random((nobs, 2))
X = sm.add_constant(X)
beta = [1, .1, .5]
e = np.random.random(nobs)
y = np.dot(X, beta) + e
# Fit regression model
results = sm.OLS(y, X).fit()
# Inspect the results
print(results.summary())


"""
OLS + dmatrices
linear rainbow
partial regression plot

We want to know whether literacy rates in the 86 French departments are associated with per capita wagers on the Royal Lottery in the 1820s
"""

import statsmodels.api as sm
#import pandas
from patsy import dmatrices

# setting up dataset
df = sm.datasets.get_rdataset("Guerry", "HistData").data
df.shape # 23 vars, 85 observation (one per department)
vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
df = df[vars]
df.tail()
df = df.dropna()
df.tail()
len(df.Department.unique()) # 85 unique departments
df.Region.unique().shape # 5 regions

# Design matrices (endog & exog)
y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')
y.head()
X.head()

# model fit and summary
mod = sm.OLS(y, X)
res = mod.fit()
print(res.summary())
res.params
dir(res)

# Diagnostics and specification tests
# Rainbow test for linearity
sm.stats.linear_rainbow(res)
# plot of partial regression for a set of regressors
sm.graphics.plot_partregress('Lottery', 'Wealth', ['Region', 'Literacy'], data=df, obs_labels=False)


"""
Fitting models using R-style formulas¶
"""

import statsmodels.formula.api as smf
import numpy as np

df = sm.datasets.get_rdataset("Guerry", "HistData").data
df = df[['Lottery', 'Literacy', 'Wealth', 'Region']].dropna()
df.head()
mod = smf.ols(formula='Lottery ~ Literacy + Wealth + Region', data=df)
res = mod.fit()
print(res.summary())
# C()
res = smf.ols(formula='Lottery ~ Literacy + Wealth + C(Region)', data=df).fit()
print(res.params)
# operator: -
res = smf.ols(formula='Lottery ~ Literacy + Wealth + C(Region) -1 ', data=df).fit()
print(res.params)
# operator: :, *
res1 = smf.ols(formula='Lottery ~ Literacy : Wealth - 1', data=df).fit()
res2 = smf.ols(formula='Lottery ~ Literacy * Wealth - 1', data=df).fit()
print(res1.params)
print(res2.params)
# vectorize functions
res = smf.ols(formula='Lottery ~ np.log(Literacy)', data=df).fit()
print(res.params)

"""
Namespaces
use parameter eval_env
default=0: caller's namespace
-1: clean namespace
"""
dir(sm)
dir(sm.graphics)
dir(sm.tsa) # time series
dir(sm.nonparametric)
dir(sm.stats)
dir(sm.emplike)
dir(sm.families)
dir(sm.genmod)

"""
generating dmatrices from formulas
(useful when the model does not accept formulas)
"""
import patsy
f = 'Lottery ~ Literacy * Wealth'
# design matrix
y, X = patsy.dmatrices(f, df, return_type='matrix')
type(y)
print(y[:5])
print(X[:5])
# data frame
y, X = patsy.dmatrices(f, df, return_type='dataframe')
type(y)
print(y[:5])
print(X[:5])
print(smf.OLS(y, X).fit().summary())















