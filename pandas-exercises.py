# -*- coding: utf-8 -*-
'''
10 Minutes to pandas
http://pandas.pydata.org/pandas-docs/stable/10min.html#min
100 Pandas Exercises (see below)
https://github.com/ajcr/100-pandas-puzzles
'''

# error in Spyder
https://github.com/spyder-ide/spyder/issues/2991
https://github.com/pydata/pandas/issues/9950
pd.get_option('display.float_format') # None
pd.reset_option('display.float_format') # set the default
pd.describe_option('display.float_format')
pd.set_option('display.float_format', lambda x:'%f'%x) # fixes this issue

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Creating a Series by passing a list of values,
# letting pandas create a default integer index:
s = pd.Series([1,3,5,np.nan,6,8])
s

# Creating a DataFrame by passing a numpy array,
# with a datetime index and labeled columns:
dates = pd.date_range('20130101', periods=6)
dates
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
df
# Creating a DataFrame by passing a dict of objects
# that can be converted to series-like.
df2 = pd.DataFrame({ 'A' : 1.,
                    'B' : pd.Timestamp('20130102'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo',
                    'G' : pd.Series(np.arange(4)),
                    'H' : np.arange(4),
                    'I' : [4,3,2,1] })
df2
df2.dtypes
df2.A
df2.B
df2.C
df2.D
df2.F
df2.G
df2.H
df2.I
# See the top & bottom rows of the frame
df.head()
df.tail()
# Display the index, columns, and the underlying numpy data
df.index
df.columns
df.values
df2.index
df2.columns
df2.values # a list of lists (=a list of rows, each row being a list of elements)
# Describe shows a quick statistic summary of your data
df.describe
df.T
df.sort_index(axis=1, ascending=False) # sort by an axis (by column labels?)

'''
Interactive work:
standard Python / Numpy expressions for selecting and setting
Production code:
the optimized pandas data access methods, .at, .iat, .loc, .iloc and .ix
'''
# selecting columns by labels, and rows by row numbers
# here: standard slices (the last excluded)
df['A'] # Selecting a single column, which yields a Series
df["A"]
df.A
df.A[dates[0]]
df[0:3] # row slices
df['20130102':'20130104']
# mixed
df.A[dates[0]]
df[0:3].A
df[0:3,'A'] # key error or type error *unhashable type 'slice'

# selection by label
# here: slices with the last included
# fast scalar selection: at()
df
dates[0]
df.loc[dates[0]] # a row selected; cross section by label
df.loc['A'] # error: row label (=index) expected
df.loc[:,['A','B']] # multi-axis by label
df.loc['20130102':'20130104',['A','B']] # both endpoints included
df.loc['20130102',['A','B']] # reduction in the dimensions of the returned object
df.loc[dates[0],'A'] # scalar value
df.at[dates[0],'A'] # fast access to a scalar

# selecting by integer position
# fast scalar selection: iat()
df.iloc[3] # select row 3 (the 4th one)
df.iloc[3:5,0:2] # by integer slices (lithe python/numpy)
df.iloc[[1,2,4],[0,2]] # by integer positions
df.iloc[1:3,:] # slicing rows explicitly
df.iloc[:,1:3] # slicing columns explicitly
df.iloc[1,1] # getting a value explicitly
df.iat[1,1] # fast access to a scalar

# Boolean Indexing
df[df.A>.5] # single column values to select rows
df[df>0] # selecting by 'where', NaNs imputed for not-selected values
df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']
df2
df2['E'].isin(['two','four']) # filtering with 'isin()'
df2[df2['E'].isin(['two','four'])]

# setting
# creating a series
s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
s1
df.at[dates[0],'A'] = 0 # set value by label: loc(), at()
df.iat[0,1] = 0 # set value by (integer) position: iloc(), iat()
df.loc[:,'D'] = 5 # set many values with a scalar
df.loc[:,'D'] = np.array([5]) # set many values with a singleton
df.loc[:,'E'] = np.array([5] * len(df)) # set many values with a numpy array (by label)
df
df.iloc[2:4,3:5] = np.arange(4).reshape(2,2) # set many values with a numpy array (by position)
df
df2 = df.copy()
df2[df2 > 0] = -df2 # where operation with setting
df2

''' missing data '''
# np.nan
# by default not included in computations

# Reindexing allows you to change/add/delete the index on a specified axis.
# This returns a copy of the data.
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1 # a copy, new dataframe with a new column E filled with NANs
df1.loc[dates[0]:dates[1],'E'] = 1
df1
df1.dropna() # complete cases only
df1.dropna(how='any') # same
df1.dropna(axis=0,how='any') # same
df1.dropna(axis='index',how='any') # same
df1.dropna(axis=1) # drop columns with NANs
df1.dropna(axis='columns') # same
df1.dropna(how='all') # rows where all values are nans
df1.loc[dates[2],'D'] = np.nan
df1
df1.dropna(thresh=4) # drop rows that have less than 4 values
df1.dropna(subset=['C','D']) # drop rows that have NANs in columns C or D
inplace=True # modify the DF, return None

?pd.DataFrame.fillna # use inplace=T if in place
df1.fillna(value=5) # fill missing data with 5
pd.isnull(df1) # boolean mask of nans

# select elements that are not null
df1[not pd.isnull(df1)] # doesn't work element-wise
not pd.isnull(df1) # doesn't work element-wise
pd.isnull(df1).apply(lambda x: x.apply(lambda x: not x)) # long, applied to each column, then to each element in the column
pd.isnull(df1).applymap(lambda x: not x) # shorter, applied to each element in each column
np.logical_not(pd.isnull(df1).as_matrix()) # using numpy
np.logical_not(pd.isnull(df1)) # the same simpler!
np.invert(pd.isnull(df1)) # using numpy
(pd.isnull(df1)*(-1)).astype('bool') # it converts into integer in between
-pd.isnull(df1) # short
~pd.isnull(df1) # short

df1[pd.isnull(df1)] # dropped elements are filled with NaN
df1[~pd.isnull(df1)]
# substitution
df1[pd.isnull(df1)] = 555 # works

''' operations '''
# Operations in general exclude missing data.
df.mean() # mean for each col
df.mean(1) # mean for each row
df.mean(0) # mean for each col

# shifting
s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
s # shifting dropps the elements at the end and introduces NaNs at the front

# subrtacting, adding, etc.
?pd.DataFrame.sub # with broadcasting
# it tries to fit subtrahends elements to minuend elements by
# column labels (default) or row indeces
# If fit is found -> subtraction is performed accross the column (row).
# If no fit -> NaNs are introduced.
df
df.sub(s, axis='index') # subtract s from columns of df (so along the index)
#   index of s fits to the index of df (both are dates)
s2 = pd.Series([1,2,3,4], index=['a','b','c','d'])
s2
df.sub(s2) # just NaNs
    # be defualt the fit is by column labels - but there's no fit
    # as df has capital letters and s2 has small letters
    # the small letters are added as new columns
s3 = pd.Series([1,2,3,4], index=['A','B','C','d'])
s3
df.sub(s3) # works OK
df.add(s3)
df.mul(s3)
df.div(s3)
df.sub(s3, axis='index') # new rows and NaNs only
    # the s2 index is letters, while df index is dates
    # no fit, new rows added to the index, NaNs introducted

# apply
df.A[0]
df.A[1]
df.A[0]+df.A[1]
np.cumsum(df.A) # can't use the axis argument of numpy for a pandas df
df.apply(np.cumsum) # apply along columns
df.apply(np.cumsum, axis=1) # apply along rows

# count
s = pd.Series(np.random.randint(0, 7, size=10))
s
s.value_counts()
np.bincount(s) # numpy counter part

# Vectorized String Methods
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s
s.str.lower()
s.str.split('_') # nothing to split, so 1-element lists returned
s.str.split('a') # see the splitting
s.str.replace('a', '?') # see replacing

# Merge
df = pd.DataFrame(np.random.randn(10, 4))
df
pieces = [df[:3], df[3:7], df[7:]]
pieces # a list of data frames
pd.concat(pieces) # back to the initial single data frame
# default: rstack(), but you can have also cstack()
pd.concat(pieces, axis=1) # concatenating by columns - but preserving the original indexing

# join
left = pd.DataFrame({'key': ['foo', 'bfoo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bfoo'], 'rval': [4, 5]})
left
right
pd.merge(left, right, on='key') # correct
pd.concat([left, right], axis=1, join='inner') # this is just stacking columns together
    # with rows of the same index put next to each other

# append
df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
df
s = df.iloc[3]
s
df.append(s, ignore_index=True) # appended at the end of df (as the last row)
    # this sets a proper index for the added row, here it's '8'
x = df.append(s, ignore_index=False) # by default doesn't check the index integrity
x   # this leaves the index as it is, here we have two rows indexed by '3'
x.loc[3] # selects two rows
x.iloc[3] # selects one row
x.loc[8] # key error (no such row label)
x.iloc[8] # selects one row
x.ix[3] # works as label location (the default) so selects two rows
x.ix[8] # error! strange, there's no row labelled 8, but there's row having position 8

''' Grouping:
Splitting the data into groups based on some criteria
Applying a function to each group independently
Combining the results into a data structure
'''
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                          'foo', 'bar', 'foo', 'foo'],
                   'B' : ['one', 'one', 'two', 'three',
                          'two', 'two', 'one', 'three'],
                   'C' : np.random.randn(8),
                   'D' : np.random.randn(8)})
df
df.groupby(['A']).sum()
df.groupby(['A','B']).sum()

''' reshaping '''

# stack
[['bar', 'bar', 'baz', 'baz',
 'foo', 'foo', 'qux', 'qux'],
['one', 'two', 'one', 'two',
 'one', 'two', 'one', 'two']]
list(zip(['bar', 'bar', 'baz', 'baz',
 'foo', 'foo', 'qux', 'qux'],
['one', 'two', 'one', 'two',
 'one', 'two', 'one', 'two']))
list(zip(*[['bar', 'bar', 'baz', 'baz', # the same
 'foo', 'foo', 'qux', 'qux'],
['one', 'two', 'one', 'two',
 'one', 'two', 'one', 'two']]))
# explanation:
 # * gets out the two sub-lists of the initial lists
 # these are passed as two arguments to zip function
 # zip creates tuples of respective pairs of elements from the two lists
 # the tuples are gathered into one list
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
index
''' This is an index consisting of two indeces, named first and second.
Each in turn is a 'factor', with given levels (bar, baz... and one, two)
and with an integer vector indicating the indeces values (the labels).'''
x = np.random.randn(8, 2)
x # 8 rows, 2 columns
df = pd.DataFrame(x, index=index, columns=['A', 'B'])
df
'''
This is a data frame with two columns A and B
and 8 rows indexed by a doubled index: (bar..., one...)
'''
# this multi indexing is hierarchical
df.loc['bar'] # DataFrame
df.loc['baz']
df.loc['one'] # error
df.loc['bar','one'] # ok
df.loc[('bar','one')] # ok
df.loc['bar'].loc['one'] # ok

df.loc['bar']['one'] # error, this asks for a column
df.loc['bar']['A'] # Series; rows (by index 'first') x column A
df.loc[:,'A'] # Series; whole column A
df.loc['bar','one']['A'] # Scalar; row x col -> a scalar, names & dims dropped

df2 = df[:4]
df2

# melting (cmp R reshape2 / tidyr)
df
stacked = df.stack() # R melt / gather
stacked
df.stack(0)
df.stack(1) # error, index has only 1 level, not 2
# casting
# R dcast / spread
stacked
stacked.unstack() # by default: the last level
stacked.unstack(0) # column 0 turned into variables/columns
stacked.unstack(1) # column 1 turned into variables/columns
stacked.unstack(2) # column 2 turned into variables/column (correct reversal)

# Pivot Tables
df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                   'B' : ['A', 'B', 'C'] * 4,
                   'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D' : np.random.randn(12),
                   'E' : np.random.randn(12)})
df
pd.pivot_table(df, values='D', index=['A','B'], columns='C')

''' Time Series
performing resampling operations during frequency conversion
(e.g., converting secondly data into 5-minutely data) '''

rng = pd.date_range('1/1/2012', periods=100, freq='S')
rng # index, time by seconds
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts # values (recorded every second)
# resampling: 100 sec = 1 min and 40 sec
ts.resample('5Min').sum() # resample into 5 min data
ts.resample('1Min').sum()
ts.resample('20S').mean() # mean values per each 20 sec periods
pd.Series.resample?

# Time zone representation
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
rng
ts = pd.Series(np.random.randn(len(rng)), rng)
ts
ts_utc = ts.tz_localize('UTC')
ts_utc

# Convert to another time zone
ts_utc.tz_convert('US/Eastern')

# time stamp vs time span
# Converting between time span representations
rng = pd.date_range('1/1/2012', periods=5, freq='M') # time stamp representation
rng # months represented as the last days of each of the 5 months
pd.date_range('1/2012', periods=5, freq='M') # the same
pd.date_range('2012', periods=5, freq='M') # the same
rng.to_period() # time span representation
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts
ps = ts.to_period()
ps # months represented as months, date & time info lost
ps.to_timestamp() # now months represented as the first day of the month

''' Converting between period and timestamp
enables some convenient arithmetic functions to be used.
In the following example,
we convert a quarterly frequency with year ending in November
to 9am of the end of the month following the quarter end:'''

# using date_range
pd.date_range('1990Q1','2000Q4', freq='Q') # dates: the last days of each period (quater), Q=Q-DEC by default
pd.date_range('1990Q1','2000Q4', freq='Q-Dec') # the same, this is the default meaning for Q
pd.date_range('1990Q1','2000Q4', freq='Q-Mar') # the same but Q=Q-MAR
pd.date_range('1990Q1','2000Q4', freq='Q-Jun') # the same but Q=Q-JUN
pd.date_range('1990Q1','2000Q4', freq='Q-Sep') # the same but Q=Q-SEP
pd.date_range('1990Q1','2000Q4', freq='Q-Nov') # shifted, the 1st Q of a year is Dec-Feb, Q=Q-NOV
pd.date_range('1990Q1','2000Q4', freq='Q-Feb') # the same but Q=Q-FEB
# using period_range
prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
prng # periods (quaters)
# converting
prng.asfreq('M') # months: the last month each period (one month for each quater in the range)
prng.asfreq('M', 'e') # the same
prng.asfreq('M', 'e') + 1 # moving by one month forward
(prng.asfreq('M', 'e') + 1).asfreq('D') # days: the last day of the last month+1 of each period
(prng.asfreq('M', 'e') + 1).asfreq('H')
# hours: the last hour, of the the last day of the last month+1 of each period
(prng.asfreq('M', 'e') + 1).asfreq('H', 's')
# time: 00am the first day of the following month - why???
(prng.asfreq('M') + 1).asfreq('H','s') # ditto
(prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
# time: 9am the first day of the following month
ts = pd.Series(np.random.random(len(prng)),prng) # time series
ts
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
ts

# categoricals
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
df
df.index
df.columns
df.values
df.describe
# checking column types
df.dtypes
df.columns.to_series().groupby(df.dtypes).groups # list of columns of a certain type
df.columns.groupby(df.dtypes) # the same shorter
# convert into categorical type
df["grade"] = df["raw_grade"].astype("category")
df["grade"] # see 3 categories
df
df.dtypes
df.values
df.columns.groupby(df.dtypes)
# Rename the categories to more meaningful names
df.grade.cat?
df.grade.cat.categories # Index(['a', 'b', 'e'], dtype='object')
df["grade"].cat.categories = ["very good", "good", "very bad"]
df.grade.cat.categories # Index(['very good', 'good', 'very bad'], dtype='object')
df
# Reorder the categories and simultaneously add the missing categories
# (methods under Series .cat return a new Series per default - but you can use "inplace")
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
df.grade.cat.categories # Index(['very bad', 'bad', 'medium', 'good', 'very good'], dtype='object')
df
df.grade # see 5 categories
# Sorting: per order in the categories, not lexical order
df.sort_values(by="grade")
df.sort_values(by="grade", ascending=False)
# df.sort('grade', ascending=False) # deprecated
# Grouping by a categorical column shows also empty categories.
df.groupby('grade').size() # all
df.groupby('grade').groups # non empty only

################# PLOTTING ####################################

# Plotting Series
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000)) # default freq is a day here
ts.head()
ts.tail()
ts2 = ts.cumsum()
ts.plot() # random noise
ts2.plot() # random walk
# If the index consists of dates, it calls gcf().autofmt_xdate() to try to format the x-axis

# Plotting Data Frames
import matplotlib.pyplot as plt
df = pd.DataFrame(np.random.randn(1000,4), index=ts.index, columns=['A','B','C','D'])
df.head() # random noise
df.tail()
df2 = df.cumsum() # random walk
plt.figure();df2.plot();plt.legend(loc='best')
df2.plot() # the same

###############################################################
# Getting Data In/Out
import os
tmp = os.getcwd()
tmp
tmp #= 'D:\\data\\Dropbox\\workarea\\python-work'
directory = 'D:\\data\\Dropbox\\workarea\\python-work\\Tutorials\\data'
if not os.path.exists(directory):
    os.makedirs(directory)
if os.getcwd() != directory:
    os.chdir(directory)
os.getcwd()


# CSV
df2.to_csv('foo.csv')
df3 = pd.read_csv('foo.csv')
# compare the two data frames (before and after CSV)
df2.head()
df3.head()
# two differences, df3 has:
# - new index column (1,2,3,...)
# - the old index as the 'Unnamed:0' column
# fixing this:
df4 = pd.read_csv('foo.csv', index_col=0)
df4.head() # great!

# HDF5
df2.to_hdf('foo.h5','df')
(pd.read_hdf('foo.h5','df')).head() # very nice!

# XLSX
df2.to_excel('foo.xlsx', sheet_name='Sheet1')
(pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])).head()

os.chdir(tmp)
################################################################



#######################################################################################





#######################################################################################


''' 100 Panadas '''

import numpy as np
import pandas as pd
pd.__version__
pd.show_versions()
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
data # columns with labels and values
labels # row labels

# 4
df = pd.DataFrame(data, labels) # 'columns' not needed as data is a dictionary and provides colnames
df
# 5
df.describe
df.info
# 6
df.iloc[:3]
df[:3]
df['a':'c']
df[:'c']
df.loc[:'c']
df.ix[:3]
df.head(3)
# 7
df[['age','animal']]
df.loc[['age','animal']] # error
df.loc[:,['age','animal']] # ok
df[[0,1]] # ok
df.iloc[:,[0,1]]
df.ix[:,['age','animal']]
df.ix[:,[0,1]]
# 8
rows=[3,4,8]
cols=['animal','age']
df[cols].iloc[rows]
df.loc[:,cols].iloc[rows]
df.iloc[rows].loc[:,cols]
df.ix[rows,cols] # best
# 9
df[[True,False]*5] # selects rows
df[:,[True,False]*2] # error, you can't select columns this way
df[df.visits>=3]
# 10
df[df.age.isnull()]
# 11
df[(df.animal=='cat') & (df.age<3)]
# 12
df[(df.age>=2) & (df.age<=4)]
# 13
df['age']['f']
df.age['f']
df['age'].f
df.age.f
df.loc['f','age']
df.ix['f','age']
df.age.f = 1.5
# 14
df.visits.sum()
# 15
df.groupby('animal').age.mean()

# 16
x = [1.1,'dog','yes',4]
# attempts
df.append(np.array(x))
# numpy array doesn't work here.
df.append(x)
# wrong: this appends a data series being a new column '0' (so 4 new rows, 1 new column)
df.append([x])
# better, but still wrong: a new row appended but index and columns have to be set.
# you need to append a data frame
pd.DataFrame([x], ['k'], df.columns)
# good!
# solution using append function
df = df.append(pd.DataFrame([x], ['k'], df.columns))
df
df = df.drop('k')
df
# solution using .loc
df.loc['k'] = x
df = df.drop('k')

# 17
df.groupby('animal').count() # too much data (shows how many non-nan values there are)
df.animal.count() # that's too little
df.animal.value_counts() # good!

# 18
df.sort_values(['age','visits'], ascending=[False,True])

# 19
# simple
df.priority = (df.priority=='yes') # very simple!
df.priority.dtype # this is bool
# official
df.priority = df.priority.map({'yes':True, 'no': False})

# 20
# simple
df.animal[df.animal=='snake'] = 'python'
# official
df.animal.replace('snake','python', inplace=True)

# 21
# For each animal type and each number of visits, find the mean age.
# In other words, each row is an animal, each column is a number of visits and the values are the mean ages 
df.pivot_table('age','animal','visits') # aggfunc = np.mean, by default

# 22
# a DataFrame df with a column 'A' of integers
df = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})
# How do you filter out rows which contain the same integer as the row immediately above

# rows which contain the same integer as the row immediately above
df[1:].values==df[:-1].values # you have to use '.values' as indeces don't match
df[df[1:].values==df[:-1].values] # Error, drop the last row or add False at the end
# adding False at the end
np.append(df[1:].values==df[:-1].values, np.array([[False]]), axis=0)
np.append(df[1:].values==df[:-1].values, False)[:,None]
# dropping the last row
df[:-1][df[1:].values==df[:-1].values]
# official solution
df.A.shift()
df.A.shift() == df.A
df[df.A.shift() == df.A]
# another one
df.shift() == df
df[(df.shift() == df).values]

# filter out rows which contain the same integer as the row immediately above
df[df.A.shift() != df.A]

# 23
# Given a DataFrame of numeric values
df = pd.DataFrame(np.random.random(size=(5, 3)))
# how do you subtract the row mean from each element in the row
df
df.mean(axis=1) # row means
df - df.mean(axis=1) # wrong: applied to columns
df.sub(df.mean(axis=1), 'rows') # done
df.sub(df.mean(axis=1), 'rows').mean(axis=1) # checking
# official
df.sub(df.mean(axis=1), axis=0) # the same

# 24
# Suppose you have DataFrame with 10 columns of real numbers, for example:
df = pd.DataFrame(np.random.random(size=(5, 10)), index=list('abcdefghij')) # wrong
df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij')) # OK
# Which column of numbers has the smallest sum? (Find that column's label.)
df.sum(axis=0).min() # minimal value, but we need index
df.sum(axis=0).idxmin() # no correct

# 25
# How do you count how many unique rows a DataFrame has
# (i.e. ignore all rows that are duplicates)?
x = pd.DataFrame(npr.randint(0,3,(10,2)), columns=list('ab'))
x
x.drop_duplicates() # unique rows
x.drop_duplicates(keep=False) # rows that do not have any duplicates
len(x.drop_duplicates(keep=False)) # just 3 rows without duplicates

# 26
# You have a DataFrame that consists of 10 columns of floating--point numbers.
# Suppose that exactly 5 entries in each row are NaN values.
# For each row of the DataFrame, find the column which contains the third NaN value.
x = pd.DataFrame(npr.randn(12,10), columns=list('abcdefghij'))
x
for i in range(12):
    x.iloc[i,npr.choice(range(10), 5, replace=False)] = np.nan
x
# loop solution - integer indexing
for i in range(12):
    c = 0
    for j in range (10):
        if np.isnan(x.iloc[i,j]):
            c += 1
            if c == 3:
                print('Row', i, 'Col', j)
                break
# loop solution - label indexing
for i in range(12):
    c = 0
    for j in list('abcdefghij'):
        if np.isnan(x.ix[i,j]):
            c += 1
            if c == 3:
                print('Row', i, 'Col', j)
                break
# done, very nice
# official
x.isnull()
x.isnull().cumsum(axis=1) # operation within columns, preserving rows
(x.isnull().cumsum(axis=1) == 3) # True indicates the correct spots
(x.isnull().cumsum(axis=1) == 3).idxmax(axis=1) # indeces (of columns) that have the 3rd value

# 27
#  A DataFrame has a column of groups 'grps' and and column of numbers 'vals'.
# For example:
df = pd.DataFrame({'grps': list('aaabbcaabcccbbc'), 
                   'vals': [12,345,3,1,45,14,4,52,54,23,235,21,57,3,87]})
# For each group, find the sum of the three greatest values.
df
df.groupby('grps').groups
pd.Series([1,3,2,4,6,5,6,6]).nlargest(3) # test
df.groupby('grps').nlargest(3) # failed
df.groupby('grps').apply(pd.Series.nlargest) # error
df.groupby('grps').apply(np.partition, 0) # failed
df.groupby('grps').apply(np.sort) # failed
df.groupby('grps')['vals'].nlargest(3) # finally got the right numbers
df.groupby('grps')['vals'].nlargest(3).sum() # sum accross groups
df.groupby('grps')['vals'].nlargest(3).sum(level=0) # sum for each group, done!
df.groupby('grps')['vals'].nlargest(3).sum(level=1) # wrong level
df.groupby('grps')['vals'].nlargest(3).unstack() # default = last level = 1
df.groupby('grps')['vals'].nlargest(3).unstack().sum(axis=1) # the same, done!
df.groupby('grps')['vals'].nlargest(3).unstack(level=0) # groups became columns
df.groupby('grps')['vals'].nlargest(3).unstack(level=0).sum() # correct
df.pivot_table('vals','grps', aggfunc='sum') # this sums up all values, not a solution

# 28
# A DataFrame has two integer columns 'A' and 'B'. The values in 'A' are between 1 and 100 (inclusive).
# For each group of 10 consecutive integers in 'A' (i.e. (0, 10], (10, 20], ...),
# calculate the sum of the corresponding values in column 'B'.

# checking
df = pd.DataFrame({'A': np.random.randint(1,101,100),
                   'B': np.random.randint(1,11,100)})
df.head()
# official
df.groupby(pd.cut(df['A'], np.arange(0, 101, 10)))['B'].sum()
# checking
pd.cut?
x = np.random.randint(1,101,100)
x
bins = np.arange(0,101,10)
bins
y = pd.cut(x, bins)
y
y.head() # error
y[0]
y[:1]
y[:2] # ok, so it's a table of labels, each indicating a range that the original x value falls into
y.isnull() # every x got a category, good
pd.cut(df['A'], np.arange(0, 101, 10)) # a Series of labels (label = a range) that can be used just like a column
df.groupby(pd.cut(df['A'], np.arange(0, 101, 10))).groups # grouping by the new labels
#   values represents indeces of rows grouped by the labels.
df.groupby(pd.cut(df['A'], np.arange(0, 101, 10)))['B'].sum() # sums of column B values, grouped by the new labels
df.B.groupby(pd.cut(df['A'], np.arange(0, 101, 10))).sum() # the same
df.B.groupby(pd.cut(df.A, np.arange(0, 101, 10))).sum() # the same

# 29
# Consider a DataFrame df where there is an integer column 'X':
df = pd.DataFrame({'X': [7, 2, 0, 3, 4, 2, 5, 0, 3, 4]})
# For each value, count the difference back to the previous zero
# (or the start of the Series, whichever is closer).
# These values should therefore be [1, 2, 0, 1, 2, 3, 4, 0, 1, 2].
# Make this a new column 'Y'.

# loop solution
df.shape
n = df.shape[0]
j=1
for i in range(n):
    if df.iloc[i,0]==0:
        j=0
    print(j)
    j+=1
# Y column
df['Y'] = 0
j=1
for i in range(n):
    if df.iloc[i,0]==0:
        j=0
    df.ix[i,'Y'] = j
    j+=1
df
df = df.drop('Y', axis=1)
# Vector solution
np.cumsum(df.X!=0) # not that easy...

# official solution 1
# it is really clever (and pure Numpy)
izero = np.r_[-1, (df['X'] == 0).nonzero()[0]] # indices of zeros
idx = np.arange(len(df)) # index for the array
df['Y'] = idx - izero[np.searchsorted(izero - 1, idx) - 1]

# checking
df.X == 0 # True if == 0
(df.X==0).nonzero() # array of indeces of positions with 0 (but inside a tuple)
(df['X'] == 0).nonzero()[0] # ditto but clean
np.r_[-1, (df['X'] == 0).nonzero()[0]] # ditto but -1 added at the beginning
izero = np.r_[-1, (df['X'] == 0).nonzero()[0]]
#
?np.searchsorted # indices where elements should be inserted to maintain order
idx = np.arange(len(df)) # index for the array
izero-1
idx
np.searchsorted(izero - 1, idx) # Array of insertion points
np.array(df.X)
# -> this is inserting idx into izero array
#   so it groups elements laying between the zeros (including the leading zeros)
#   so 1 (series 1), 2 (series 2), 3 (series 3)
#   so these are values replaced with position (+1) of their leading zero in izero array
np.searchsorted(izero - 1, idx) - 1
# -> ditto but shifted to (0,1,2) from (1,2,3)
izero[np.searchsorted(izero - 1, idx) - 1]
# -> the values are replaced with positions (indeces) of their leading zeros (ie. -1, 2, 7)
idx - izero[np.searchsorted(izero - 1, idx) - 1]
# -> here just deducting the leading zeros positions - done
#    it gives positions relative to the leading zero for each series

# official solution 2
# this is also clever
x = (df['X'] != 0).cumsum()
y = x != x.shift()
df['Y'] = y.groupby((y != y.shift()).cumsum()).cumsum()
# checking
df['X'] != 0
x = (df['X'] != 0).cumsum()
x
y = x != x.shift()
y
y != y.shift()
(y != y.shift()).cumsum()
y.groupby((y != y.shift()).cumsum()).groups
y.groupby((y != y.shift()).cumsum()).cumsum()

# 30
# Consider a DataFrame containing rows and columns of purely numerical data.
# Create a list of the row-column index locations of the 3 largest values.
x = np.random.randint(0,20,(4,5))
df = pd.DataFrame(x, columns=list('abcde'))
df
df.idxmax() # max per column
# solution
df.unstack().sort_values(ascending=False)[:3].index.tolist()
# official solution
df.unstack().sort_values()[-3:].index.tolist()
# checking
df
df.unstack()
df.unstack().sort_values()
df.unstack().sort_values()[-3:]
df.unstack().sort_values()[-3:].index
df.unstack().sort_values()[-3:].index.tolist()

# 31
# Given a DataFrame with a column of group IDs, 'grps',
# and a column of corresponding integer values, 'vals',
# replace any negative values in 'vals' with the group mean.
df = pd.DataFrame({'grps':np.random.choice(list('abcd'),10),
                   'vals':np.random.randint(-50,50,10)})
df
df[df.vals>=0].groupby('grps').vals.mean()

# solution 1: mean of the positive values only
df.ix[df.vals<0,'vals'] = np.nan
tmp1 = df.groupby('grps').apply(lambda x: x.fillna(np.mean(x))) # inplace=True doesn't work here
tmp1
# -> the result is a hierarchical Data Frame
#    constructed from stacked series, each serie for one group
#    You need to unstack it and preserve just the number index
# dropping the unneeded level in the index:
# method 1:
df['vals'] = tmp1.reset_index(0, drop=True).vals
df

# method 2 and 3 (instead of method 1)
tmp1.index = tmp1.index.get_level_values(1)
tmp1.index = tmp1.index.droplevel(0) # index is not sorted but it's not a problem
df['vals'] = tmp1.vals # assignement is correct per indexing
df

# solution 2: ditto
df.ix[df.vals<0,'vals'] = np.nan
data = df.groupby('grps').vals # SeriesGroupBy
data.size()
data.count()
tmp2 = data.transform(lambda x: x.fillna(np.mean(x)))
tmp2
# -> here you get a Series, that you can substiture for a column in your data frame
df['vals'] = tmp2 # index is sorted, all fits well
df

# Extra:
# tmp2 = data.transform(lambda x: 0 if np.all(pd.isnull(x)) else x.fillna(np.mean(x)))
# -> in case only negative values: 0 on NaN?

# official solution: ditto
def replace(group):
    mask = group<0
    group[mask] = group[~mask].mean()
    return group
df.groupby(['grps'])['vals'].transform(replace)

# checking
df.groupby(['grps'])['vals'].groups
def replace(group):
    print(type(group)) # Series
    print(group)
    j=0
    for i in group:
        print(j, type(i), i) # int, Series, 0..n
        j+=1
    return group
df.groupby(['grps'])['vals'].transform(replace)

# short versions:
df = pd.DataFrame({'grps':np.random.choice(list('abcd'),10),
                   'vals':np.random.randint(-50,50,10)})
dfcopy = df.copy()
df
df[df.vals>=0].groupby('grps').vals.mean()
# 1
df.ix[df.vals<0,'vals'] = np.nan
df.groupby('grps').apply(lambda x: x.fillna(np.mean(x))).reset_index(0, drop=True).vals
# 2
df.ix[df.vals<0,'vals'] = np.nan
df.groupby('grps').vals.transform(lambda x: x.fillna(np.mean(x)))


# 32
# Implement a rolling mean over groups with window size 3, which ignores NaN
# value. For example consider the following DataFrame:
df = pd.DataFrame({'group': list('aabbabbbabab'),
                   'value': [1, 2, 3, np.nan, 2, 3,np.nan, 1, 7, 3, np.nan, 8]})
df
# The idea is to sum the values in the window (using sum), count the NaN
# values (using count) and then divide to find the mean.
g1 = df.groupby('group').value # group values
g1.groups
g2 = df.fillna(0).groupby('group').value # fillna, then group values
g2.groups
g2.value_counts() # it has NaN's replaced with 0
g2.head
df.ix[g2.groups['a']] # but ...df is still filled with NaNs
df.ix[g2.groups['b']]
df # NaNs are still present
s = g2.rolling(3, min_periods=1).sum() / g1.rolling(3, min_periods=1).count() # compute means
s
s.reset_index(level=0, drop=True).sort_index()  # drop/sort index

# checking
df.groupby('group').value.rolling(3,1).count()
df.fillna(0).groupby('group').value.rolling(3,1).sum()
# -> if not minimal windows=1, then NaN's introduced
df.fillna(0).groupby('group').value.rolling(3,1).sum() / \
    df.groupby('group').value.rolling(3,1).count()

"""
Series and DatetimeIndex
"""

# 33
# Create a DatetimeIndex that contains each business day of 2015 and use it
# to index a Series of random numbers. Let's call this Series s.
index = pd.date_range('2015','2016', freq='B')[:-1] # business day frequency
index
n = index.size # = shape[0]
n
s = pd.Series(np.random.random(n), index)
s
# official
dti = pd.date_range(start='2015-01-01', end='2015-12-31', freq='B')
s = pd.Series(np.random.rand(len(dti)), index=dti)

# 34
# Find the sum of the values in s for every Wednesday.
s.head()
s.index
s.index.weekday
s[s.index.weekday==2].sum()
s.index.month
s.index.year


# 35
# For each calendar month in s, find the mean of values.
s.groupby(s.index.month).mean()
# official
s.resample('M').mean()

# 36
# For each group of four consecutive calendar months in s, find the date on which the highest value occurred
# 1, failed
s.resample('Q').head() # these are quater averages
s.resample('Q').max() # almost - these are max values withing each quater, but you don't know exact dates on when these values occured; you just have (quater, max value) pairs
s.resample('Q').idxmax() # no - this returns a quater that has maximal average value
s.resample('Q').mean().idxmax() # ditto but without a warning
s.resample('Q').argmax() # ditto
# 2, done
s.groupby(s.index.quarter).max() # almost - max values for each quater, but no exact dates
s.groupby(s.index.quarter).idxmax() # done!
s[s.groupby(s.index.quarter).idxmax()] # checking, done!
# 3, official
s.groupby(pd.TimeGrouper('4M')).idxmax()
# -> best as it also indicates the quaters exactly

# 37
# Create a DateTimeIndex consisting of the third Thursday in each month for the years 2015 and 2016.
pd.date_range('2015','2016', freq='W-THU') # weekly, every Thursday
# official
pd.date_range('2015-01-01', '2016-12-31', freq='WOM-3THU')
# -> now, try to figure it out yourself
pd.date_range('2015','2016', freq='W-THU')
pd.date_range('2016','2017', freq='WOM-1FRI')
# -> the first Friday in each month in 2017


"""
Cleaning data
"""

df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm', 
                               'Budapest_PaRis', 'Brussels_londOn'],
              'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
              'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )', 
                               '12. Air France', '"Swiss Air"']})
df

# 38
# Some values in the the FlightNumber column are missing. These numbers are meant to increase by 10 with each row so 10055 and 10075 need to be put in place. Fill in these missing numbers and make the column an integer column (instead of a float column)
df.FlightNumber.interpolate().astype('int')

# 39
# The From_To column would be better as two separate columns! Split each string on the underscore delimiter _ to give a new temporary DataFrame with the correct values. Assign the correct column names to this temporary DataFrame.
df.From_To
# 1
df.From_To.str.split('_') # a series of 2-elem. lists
pd.DataFrame(df.From_To.str.split('_')) # no
df.From_To.str.split('_').tolist() # an array build using nested lists
pd.DataFrame(df.From_To.str.split('_').tolist(), columns=['From','To']) # done!
# 2
df.From_To.str.split('_').apply(lambda x: print(x))
df.From_To.str.split('_').apply(pd.Series) # this works, don't know how actually
# 3
#df.From_To.str.split('_', 1, expand=True) # works! (this limits the splits to 1)
df.From_To.str.split('_', expand=True) # works!
# 4
df.From_To.str.extract('([A-Za-z]+)', expand=True)
df.From_To.str.extract('([A-Za-z]+)(\w+)', expand=True)
df.From_To.str.extract('(?P<From>[A-Za-z]+)(?P<To>\w+)', expand=True)
df.From_To.str.extract('(?P<From>[a-zA-Z]+)_(?P<To>[a-zA-Z]+)', expand=True) # done!
# official
temp = df.From_To.str.split('_', expand=True)
temp.columns = ['From', 'To']
temp

# 40
# Notice how the capitalisation of the city names is all mixed up in this temporary DataFrame. Standardise the strings so that only the first letter is uppercase (e.g. "londON" should become "London".)
# 1 official
temp.From.str.capitalize()
temp.To.str.capitalize()
# 2 more general
temp.apply(lambda x: x.apply(str.capitalize))
# -> you need double apply here, as you apply this to elements, not vectors. One apply breaks down the DataFrame into series, the other apply the function to series elements.
temp.apply(lambda x: x.str.capitalize())
# -> nice, short
temp.applymap(str.capitalize) # best solution!

# >>> updating df
df['FlightNumber'] = df.FlightNumber.interpolate().astype('int') # you could use 'inplace' too
temp = df.From_To.str.split('_', expand=True)
temp.columns = ['From', 'To']
temp = temp.applymap(str.capitalize)

# 41
# Delete the From_To column from df and attach the temporary DataFrame from the previous questions
df.drop('From_To', axis=1, inplace=True)
pd.concat([df, temp], axis=1) # the same
df = df.join(temp)

# 42
# In the Airline column, you can see some extra puctuation and symbols have appeared around the airline names. Pull out just the airline name. E.g. '(British Airways. )' should become 'British Airways'.
df.Airline
df.Airline.str.replace('[^ a-zA-Z]', '').str.strip() # works!
df.Airline.str.replace('[^ a-zA-Z]', '').str.replace('^\s+|\s+$', '') # works!
df.Airline.str.replace('[^ a-zA-Z]|\s+$', '') # no idea why this doesn't work
# official
df.Airline.str.extract('([a-zA-Z\s]+)', expand=False) # by default: expand=False (return a Series) but warning: default will be changed
df.Airline.str.extract('([a-zA-Z\s]+)', expand=False).str.strip()
df['Airline'] = df.Airline.str.replace('[^ a-zA-Z]', '').str.strip()

# 43
# In the RecentDelays column, the values have been entered into the DataFrame as a list. We would like each first value in its own column, each second value in its own column, and so on. If there isn't an Nth value, the value should be NaN.
# Expand the Series of lists into a DataFrame named delays, rename the columns delay_1, delay_2, etc. and replace the unwanted RecentDelays column in df with delays.
# attempts
df.RecentDelays
df.RecentDelays.apply(pd.Series) # each list turned into a Series, the result is a DataFrame
# best
df.RecentDelays.apply(pd.Series).rename(columns=lambda x: 'delay_' + str(x+1)) # done
# the same using a dictionary
delays = df.RecentDelays.apply(pd.Series)
old = delays.columns
d = dict(zip(old, ['delay_'+str(x+1) for x in old]))
delays.rename(columns = d)
# official
delays = df['RecentDelays'].apply(pd.Series)
delays.columns = ['delay_{}'.format(n) for n in range(1, len(delays.columns)+1)]
# modify the dataframe
delays = df.RecentDelays.apply(pd.Series).rename(columns=lambda x: 'delay_' + str(x+1))
df = df.drop('RecentDelays', axis=1).join(delays)

# 38 - 43
#(extract of the operations)
df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm', 
                               'Budapest_PaRis', 'Brussels_londOn'],
              'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
              'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )', 
                               '12. Air France', '"Swiss Air"']})
df
df['FlightNumber'] = df.FlightNumber.interpolate().astype('int') # you could use 'inplace' too
temp = df.From_To.str.split('_', expand=True)
temp.columns = ['From', 'To']
temp = temp.applymap(str.capitalize)
df = df.drop('From_To', axis=1)
df = df.join(temp)
df['Airline'] = df.Airline.str.replace('[^ a-zA-Z]', '').str.strip()
delays = df.RecentDelays.apply(pd.Series).rename(columns=lambda x: 'delay_' + str(x+1))
df = df.drop('RecentDelays', axis=1).join(delays)
df

"""
Using MultiIndexes
"""

# 44
# Given the lists letters = ['A', 'B', 'C'] and numbers = list(range(10)), construct a MultiIndex object from the product of the two lists. Use it to index a Series of random numbers. Call this Series s.
# solution 1
letters = ['A', 'B', 'C'] # list('ABC')
numbers = list(range(10))
index = pd.MultiIndex.from_product([letters, numbers]) #, names=['A','B'])
s = pd.Series(data=np.random.random(index.size), index=index)
s # done!

# 45
# Check the index of s is lexicographically sorted (this is a necessary property for indexing to work correctly with a MultiIndex).
# official
s.index.is_lexsorted()
# or more verbosely...
s.index.lexsort_depth == s.index.nlevels
s.index.lexsort_depth
s.index.nlevels

# 46
# Select the labels 1, 3 and 6 from the second level of the MultiIndexed Series.
s[1:3] # just rows 1,2
s['A'] # all under A
s.xs('A')
s.A[1] # A-1
s['A',1]
s.ix['A',1]
s.xs(('A',1))
s.xs(1, level=1) # all under 1, level 1 (so from A,B,C)
# check the levels
s.index.get_level_values(0).unique()
s.index.get_level_values(1).unique()
# select labels from the second level (level=1)
s.xs(1, level=1) # OK, now get the values for 3 labels, not just one
s.xs(3, level=1)
# official
s.loc[:, [1,3,6]] # works!
# -> this is the same as if [1,3,6] where columns in a dataframe
# checking
df = s.unstack() # =1
df
df.loc[:,[1,3,6]] # so
s.unstack().loc[:,[1,3,6]].stack() # the same
s.unstack()[[1,3,6]].stack() # the same

# 47
# Slice the Series s; slice up to label 'B' for the first level and from label 5 onwards for the second level.
s
s.loc[:'B', 5:]
# official
s.loc[pd.IndexSlice[:'B', 5:]]
s.loc[slice(None, 'B'), slice(5, None)]
# checking
slice(None, 'B')

# 48
# Sum the values in s for each label in the first level (you should have Series giving you a total for labels A, B and C).
# sums under A, B, C
s.unstack(0)
s.unstack(0).sum()
s.sum(level=0)

# 49
# Suppose that sum() (and other methods) did not accept a level keyword argument. How else could you perform the equivalent of s.sum(level=1)?
s.unstack(0).sum()

# 50
# Exchange the levels of the MultiIndex so we have an index of the form (letters, numbers). Is this new Series properly lexsorted? If not, sort it.
s.index.swaplevel(1, 0)
s2 = s.copy()
s2.index = s2.index.swaplevel(1, 0)
s2
s2.index.is_lexsorted() # False
s2.sort_index()
# solution
s.index = s.index.swaplevel(1, 0)
s.index.is_lexsorted()
s.sort_index(inplace=True)
s
# official
new_s = s.swaplevel(0, 1)
# check
new_s.index.is_lexsorted()
# sort
new_s = new_s.sort_index()

# 51
# Minesweeper: generate 5x4 grid
X = 5
Y = 4
x = np.arange(X)
y = np.arange(Y)
# 1
from itertools import product
list(product(x,y))
pd.DataFrame(list(product(x,y)), columns=['x','y']) # done
# 2
np.meshgrid(x,y)
np.dstack(np.meshgrid(x,y))
np.dstack(np.meshgrid(x,y)).reshape(-1,2)
tmp = np.dstack(np.meshgrid(x,y)).reshape(-1,2)
pd.DataFrame(tmp, columns=['y','x']) # done
# 3
from sklearn.utils.extmath import cartesian
pd.DataFrame(cartesian([x,y]), columns=['x','y']) # done
# 4
tmp = pd.tools.util.cartesian_product([x,y])
pd.DataFrame(np.transpose(tmp), columns=list('xy')) # done
# official
p = pd.tools.util.cartesian_product([x, y])
df = pd.DataFrame(np.asarray(p).T, columns=['x', 'y'])

# 52
# For this DataFrame df, create a new column of zeros (safe) and ones (mine). The probability of a mine occuring at each location should be 0.4.
n = len(df.index)
from scipy.stats import bernoulli
v = bernoulli(0.4).rvs(n)
bernoulli.rvs(0.4, size=n) # alternative
from scipy.stats import binom
binom(1, 0.4).rvs(n) # alternative
binom.rvs(1, 0.4, size=n) # alternative
df = df.assign(mine=v)
df
df.mine.value_counts()
df.mine.mean()
# official
df['mine'] = np.random.binomial(1, 0.4, X*Y)
# checking
np.random.binomial(1, 0.4, X*Y)

# 53
# Now create a new column for this DataFrame called 'adjacent'. This column should contain the number of mines found on adjacent squares in the grid.
# (E.g. for the first row, which is the entry for the coordinate (0, 0), count how many mines are found on the coordinates (0, 1), (1, 0) and (1, 1).)

# first build loop solution
# next try some vectorized or pandas

# <---------------------------------------------------------------------------------






























# create df
df = pd.DataFrame(data=data, index=labels)
df = pd.DataFrame(data, labels) # the same
df

# R.str
df.info() # enough for small arrays, inculdes dispaly size limits
# summarydf.index
df.columns
df.values
df.describe() # only numerical
df.describe(include='all') # all
df.describe(include=[type]) # only categorical
df.describe(include=[np.number]) # only numerical (like default None)

# view data
df.head(3)
df[:3]
df.iloc[:3]
df.iloc[:3,]
df.iloc[:3,:]
df[:'c']
df.loc[:'c']
df[range()]



# interactive tools: [] and .

# [] to get rows
df['a'] # error
df['a':'a'] # row 'a'
df['a':'b'] # rows a and b (0 and 1)     <---------------
df['a':'b',] # error
df[['a']] # error
df[0] # error
df[0:0] # empty frame
df[0:1] # row 0                          <---------------
df[0:1,] # error
df[[0]] # column 0
df[[0,1]] # column 0 and 1
df[[0:1]] # error
# [] to get cols
df['age'] # col age (0)                  <---------------
df['age':'animal'] # empty frame (no rows selected)
df[:,'age'] # error
df[:,'age':'animal'] # error
df[['age']] # col age (0)
df[['age','animal']] # age and animal cols
df[['age':'animal']] # error
df[:,0]# error
df[:,0:1] # error
df[[0]] # column 0
df[[0,1]] # column 0 and 1
df[[0:1]] # error
# . to get cols only
df.age

# [] with boolean indeces - for rows only
df[[True, False]*5] # selected rows
df[np.arange(df.shape[0])<3] # first 3 rows by boolean


# so you can get only:
# a slice of rows by label (last element included)
df['b':'d']
# a slice of rows by position (last element exluded)
df[1:3]
# a single column by label
df['priority']
df.priority
# a selection of columns by a list of labels or positions
df[['age','animal']]
df[[0,1]]


# you can mix/connect them:
df['b':'d'].age
df['age'][:2]

# iloc
df.iloc[0] # row 0
df.iloc[0,] # row 0
df.iloc[:,0] # col 0

# loc
df.loc['a'] # row 'a' (0)
df.loc['a',] # row 'a' (0)
df.loc['age'] # error, no such row
df.loc[:,'age'] # column age (0)

# geting scalars only
df.at # label location
df.iat # integer location

# super mix all-in-one:
    # to obtain both scalars and sub-arrays
    # search by label by default with a fallback to integer location
df.ix



# format numeric values for display
df = pd.DataFrame(np.random.randn(3, 3))
df
df = df.applymap(lambda x: '%.2f' % x)
df

# get the number of rows
df.shape
df.shape[0] # rows

# Select just the 'animal' and 'age' columns
df[['age','animal']]
pd.concat([df.age, df.animal], axis=1)
df.loc[:,'age':'animal']
df.loc[:,['age','animal']]
df.iloc[:,0:2]
df.iloc[:,[0,1]]

# Select the data in rows [3, 4, 8] and in columns ['animal', 'age'].
df[['age','animal']][3:5] # can have only a slice of rows
pd.concat([df[['age','animal']][3:5],df[['age','animal']][8:9]])
df[['age','animal']].iloc[[3,4,8]]
df[['age','animal']].loc[['d','e','i']]
df.iloc[[3,4,8],0:2]
df.loc[['d','e','i'],['age','animal']]
df.iloc[[3,4,8]].loc[:,['age','animal']]
df.loc[:,['age','animal']].iloc[[3,4,8]]
df.ix[[3, 4, 8], ['animal', 'age']] # label -> integer, this works nice

    # errors (revisiting the scalar subsetting functions)
df.at[[3, 4, 8], ['animal', 'age']] # error label/integer
df.iat[[3, 4, 8], ['animal', 'age']] # error label/integer
df.iat[[3, 4, 8]].at[:,['animal', 'age']] # error - this is only for scalar
df.iat[3,].at[:,'animal'] # error - this is only for scalar, while at tries to return whole column

# Select the rows where the number of visits is greater than 2
df[df.visits > 2] # boolean indexing works for rows only

# Select the rows where the age is missing, i.e. is NaN
    # exploring
df.isnull() # bool for each element
df.notnull() # bool for each element
df is None # false, this checks the object itself
None == None # True
np.nan == np.nan # False
    # homing in on
df.dropna() # complete cases only, which by accident is fine here
df.dropna(subset=['age']) # drop cases where age is missing
df[df.age.notnull()] # equivalent, works for one column only
    # but we should select these rows, not exclude them
df[df.age.isnull()]

# Select the rows where the animal is a cat and the age is less than 3.
df[df.animal=="cat"]
df[df.age<3]
# how to connect these?
df.animal=="cat" & df.age<3 # error
type(df.animal=="cat") # Series;
(df.animal=="cat") & (df.age<3) # works OK
df[(df.animal=="cat") & (df.age<3)]

# Select the rows the age is between 2 and 4 (inclusive).
df[(df.age >=2) & (df.age <= 4)]
df[df.age.between(2,4)] # between works for Series
df.query('2<=age<=4')

# Change the age in row 'f' to 1.5
df
df.age['f'] = 1.5
df['f':'f'].age = 1.6
df.loc['f','age'] = 1.7
df.at['f','age'] = 1.8
df.ix['f','age'] = 1.9
df

# Calculate the sum of all visits
df.visits.sum()

# Calculate the mean age for each different animal in df
df.groupby('animal').age.mean()

# Append a new row 'k' to df with your choice of values for each column. Now delete that row.
    # attempt 1
new_row = df = pd.DataFrame(data={'age':5.5,'animal':'snake','priority':'no','visits':3},
                            index=['k'])
new_row
df = df.append(new_row)
df['k':'k']
df.loc['k']
df.ix['k']
df[df.index != 'k']
df = df[df.index != 'k']
df
    # official
df.loc['k'] = [5.5, 'dog', 'no', 2] # nice!
df
df.drop('k') # nice!
df = df.drop('k') # to save the result
df.drop('k', inplace=True) # inplace, None returned

# Count the number of each type of animal in df
len(df.animal.unique()) # no, this is how many unique values there are
df.animal.value_counts()
df.groupby('animal').animal.count()

# Sort df first by the values in the 'age' in decending order,
# then by the value in the 'visit' column in ascending order.
df.sort(['age','visits'], ascending=[False,True]) # DEPRECATED
df.sort_values(by=['age', 'visits'], ascending=[False, True])

# The 'priority' column contains the values 'yes' and 'no'.
# Replace this column with a column of boolean values:
# 'yes' should be True and 'no' should be False.
df.dtypes
df.priority = (df.priority == 'yes') # simple test and assignement
df
df.dtypes
# official - dictionary mapping
df.priority = ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']
df
df['priority'] = df['priority'].map({'yes': True, 'no': False})
df

# In the 'animal' column, change the 'snake' entries to 'python'.
df.animal[df.animal=='snake'] = 'python' # warnings
df.animal.replace('snake','python') # official; not in place
df.animal = df.animal.replace('snake','python')
df.animal.replace('snake','python', inplace=True)

# For each animal type and each number of visits,
# find the mean age (hint: use a pivot table).
df.groupby(['animal','visits']).age.mean()
df.pivot_table(values='age',columns=['animal','visits'],aggfunc=np.mean)
df.pivot_table(values='age',columns=['animal','visits']) # default: aggfunc=np.mean
df.pivot_table(values='age',index='visits',columns='animal') # default: aggfunc=np.mean # wide table (melted)

# You have a DataFrame df with a column 'A' of integers. How do you filter out rows
# which contain the same integer as the row immediately above?
    # 1, works
df2 = pd.DataFrame({'A':np.random.randint(5,size=20)})
df2.A==df2.A
df2
df2.A[1:]
df2.A[:-1]==df2.A[1:]
df2[1:][(df2.A[:-1]==df2.A[1:]).values] # works!
    # 2 (reformulation)
df2.loc[(df2.A[:-1]==df2.A[1:]).values] # wrong indexing of rows (index is decreased/too low by 1)
# No, I should filter out these, so:
    # official
df2.loc[df2['A'].shift() != df2['A']]
    # checking
df2
df2.shift() # NaN introducted at position 0; values changed to float64;
np.NaN != 2 # True

# Given a DataFrame of numeric values, how do you subtract the row mean
# from each element in the row?
df = pd.DataFrame({'A': np.random.randint(0,10,20),
                   'B': np.random.randint(20,40,20)})
df
df.mean() # mean per each column
df.mean(axis=0) # mean per each column
df.mean(axis=1) # mean per each row
df - df.mean(axis=1) # wrong, NaNs
df.subtract(df.mean(axis=1), axis=0) # works
df.sub(df.mean(axis=1), axis=0) # same

# Suppose you have DataFrame with 10 columns of real numbers
# Which column of numbers has the smallest sum? (Find that column's label.)
df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
df.sum(0) # sum for every column # this is the default axis
df.sum(0).argmin() # column of the smallest sum
df.sum().idxmin() # official, the same
# checking
df
df.idxmin(axis=0) # default: index of the row, for each column
df.idxmin(axis=1) # column label, for each row

# How do you count how many unique rows a DataFrame has
# (i.e. ignore all rows that are duplicates)?
df = pd.DataFrame(np.random.randint(0,2,(10,3)), columns=list('abc'))
df
np.all(df.ix[0] == df.ix[1])
#df.irow(0)==df.irow(2) # deprecated
# loop solution: unique rows
s=0
for i in np.arange(df.shape[0]):
    u = True
    for j in np.arange(i):
        if np.all(df.iloc[i]==df.iloc[j]):
            u = False
            print(i, 'is the same as', j)
            break
    if u:
        print(i, 'is unique')
        s += 1
print('Number of unique rows is', s)
# Series has a nunique() and unique() methods
df.a.nunique()
# DataFrame: duplicated()
df.duplicated() # first unique rows are False, all else is True
~df.duplicated() # marking True all the unique rows
(~df.duplicated()).sum() # the sum
# all rows that don't have duplicates
df.duplicated(keep=False)
# solutions:
(~df.duplicated(keep=False)).sum() # 1
len(df) - df.duplicated(keep=False).sum() # 2
# DataFrame: drop_duplicates()
len(df.drop_duplicates(keep=False)) # 3


''' You have a DataFrame that consists of 10 columns of floating--point numbers.
Suppose that exactly 5 entries in each row are NaN values.
For each row of the DataFrame, find the column which contains the third NaN value. '''

df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
df
#for i in np.arange(df.shape[1]): # 3 nan's in every column...
#    for j in np.random.choice(df.shape[0], 3, replace=False):
#        df.iloc[j,i] = np.nan
#df
for i in np.arange(df.shape[0]): # 5 nan's in every row (now correct)
    for j in np.random.choice(df.shape[1], 5, replace=False):
        df.iloc[i,j] = np.nan
df
df.isnull()
# 1, loop solution
for i in np.arange(df.shape[0]):
    s = 0
    for j in np.arange(len(df.iloc[i])):
        if np.isnan(df.iloc[i,j]):
            s += 1
        if s == 3:
            print('nan in', i, 'is at', df.columns[j])
            break
# official
(df.isnull().cumsum(axis=1) == 3).idxmax(axis=1)
# checking
df.isnull()
df.isnull().cumsum() # default: by columns, we want by rows
df.isnull().cumsum(axis=1) # now by rows
df.isnull().cumsum(axis=1) == 3 # find the first occurance of 3
(df.isnull().cumsum(axis=1) == 3).idxmax(axis='columns') # now find the index (column)

''' A DataFrame has a column of groups 'grps' and and column of numbers 'vals'.
For each group, find the sum of the three greatest values. '''

df = pd.DataFrame({'grps': list('acbacacababcabacacaabaabbcabac'),
                   'vals': np.random.randint(0,20,30)})
df[df.grps == 'a']
df[df.grps == 'b']
df[df.grps == 'c']
df.groupby('grps').groups # DataFrame groupby object
df.groupby('grps')['vals'].groups # breaking down into Series GroupBy objects
df.groupby('grps')['vals'].apply(np.sort)
df.groupby('grps')['vals'].apply(np.partition, kth=-3) # faster
# -> now just sum the last 3 elements
x = np.random.choice(np.arange(10), 10, replace=False)
#x = np.array([1,2,3,4,5])
x[-3:].sum()
x.take((-3,-2,-1))
pd.DataFrame(x).nlargest(2, 0)
df.groupby('grps')['vals'].apply(np.partition, kth=-3).apply(np.take, indices=(-3,-2,-1)) # last 3 elements
# solution
df.groupby('grps')['vals'].apply(np.partition, kth=-3).apply(np.take, indices=(-3,-2,-1)).apply(sum) # done
# official 1
df = df.sort_values('vals', ascending=False) # this is not needed
df.groupby('grps')['vals'].nlargest(3).sum(level=0)
# checking 
df.groupby('grps')['vals'].nlargest(3) # pd.Series method - see, this is a hierarchical dataframe,
                # level=0 are groups, level=1 is meaningless
df.groupby('grps')['vals'].nlargest(3).sum() # wrong, sum over all elements
df.groupby('grps')['vals'].nlargest(3).sum(level=0) # for hierarchical pd.DF indices: level 0 (per)
df.groupby('grps')['vals'].nlargest(3).sum(level=1) # now: the meaningless grouping sum
# official, with unstacking (so you sum over an axis instead a level)
df.groupby('grps')['vals'].nlargest(3).unstack().sum(axis=1)
# official 2
df.groupby('grps')['vals'].apply(lambda x: x.nlargest(3).sum())

''' A DataFrame has two integer columns 'A' and 'B'. The values in 'A' are between 1 and 100 (inclusive).
 For each group of 10 consecutive integers in 'A' (i.e. (0, 10], (10, 20], ...),
 calculate the sum of the corresponding values in column 'B'. '''

df = pd.DataFrame({'A': np.random.randint(1,101,100),'B':np.random.randint(0,20,100)})
df = pd.DataFrame({'A': np.arange(1,101),'B':np.random.randint(0,20,100)})
df



# repeat pandas

# add new column, A mod 10
# group by the new column
# sum























''' FULL PANDAS '''

# appending
df1 = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
df1
df2
df1.append(df2)

# mean
d = pd.DataFrame({'g':('a','b'),'v':(-1,np.nan)})
d
d.groupby('g').mean()


# Hierarchical Index / Columns, Multi index
# Dropping a level, see:
# 31
# Given a DataFrame with a column of group IDs, 'grps',
# and a column of corresponding integer values, 'vals',
# replace any negative values in 'vals' with the group mean.


# Pivot tables
import datetime
df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 6,
                   'B': ['A', 'B', 'C'] * 8,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 4,
                   'D': np.random.randn(24),
                   'E': np.random.randn(24),
                   'F': [datetime.datetime(2013, i, 1) for i in range(1, 13)] +
                        [datetime.datetime(2013, i, 15) for i in range(1, 13)]})
df
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
pd.pivot_table(df, values='D', index=['B'], columns=['A', 'C'], aggfunc=np.sum)
pd.pivot_table(df, values=['D','E'], index=['B'], columns=['A', 'C'], aggfunc=np.sum)
pd.pivot_table(df, values='D', index=pd.Grouper(freq='M', key='F'), columns='C')
# showing NANs as ""
table = pd.pivot_table(df, index=['A', 'B'], columns=['C'])
print(table.to_string(na_rep=''))
# adding subtotals and totals, or rather
# adding subtotals for columns and totals for rows
df.pivot_table(index=['A', 'B'], columns='C', margins=True, aggfunc=np.std)
df
df.pivot_table(index=['A','B','C'], values = ['D','E'], aggfunc=np.sum, margins=True)
df.pivot_table(index=['A','B'], columns=['C'], values = ['D','E'], aggfunc=np.sum, margins=True)



"""
Time Series
http://pandas.pydata.org/pandas-docs/stable/timeseries.html

Offset
Alias	Description
B	business day frequency
C	custom business day frequency (experimental)
D	calendar day frequency
W	weekly frequency
M	month end frequency
SM	semi-month end frequency (15th and end of month)
BM	business month end frequency
CBM	custom business month end frequency
MS	month start frequency
SMS	semi-month start frequency (1st and 15th)
BMS	business month start frequency
CBMS	custom business month start frequency
Q	quarter end frequency
BQ	business quarter endfrequency
QS	quarter start frequency
BQS	business quarter start frequency
A	year end frequency
BA	business year end frequency
AS	year start frequency
BAS	business year start frequency
BH	business hour frequency
H	hourly frequency
T, min	minutely frequency
S	secondly frequency
L, ms	milliseconds
U, us	microseconds
N	nanoseconds

Anchored:
Alias	Description
W-SUN	weekly frequency (sundays). Same as ‘W’
W-MON	weekly frequency (mondays)
W-TUE	weekly frequency (tuesdays)
W-WED	weekly frequency (wednesdays)
W-THU	weekly frequency (thursdays)
W-FRI	weekly frequency (fridays)
W-SAT	weekly frequency (saturdays)
(B)Q(S)-DEC	quarterly frequency, year ends in December. Same as ‘Q’
(B)Q(S)-JAN	quarterly frequency, year ends in January
(B)Q(S)-FEB	quarterly frequency, year ends in February
(B)Q(S)-MAR	quarterly frequency, year ends in March
(B)Q(S)-APR	quarterly frequency, year ends in April
(B)Q(S)-MAY	quarterly frequency, year ends in May
(B)Q(S)-JUN	quarterly frequency, year ends in June
(B)Q(S)-JUL	quarterly frequency, year ends in July
(B)Q(S)-AUG	quarterly frequency, year ends in August
(B)Q(S)-SEP	quarterly frequency, year ends in September
(B)Q(S)-OCT	quarterly frequency, year ends in October
(B)Q(S)-NOV	quarterly frequency, year ends in November
(B)A(S)-DEC	annual frequency, anchored end of December. Same as ‘A’
(B)A(S)-JAN	annual frequency, anchored end of January
(B)A(S)-FEB	annual frequency, anchored end of February
(B)A(S)-MAR	annual frequency, anchored end of March
(B)A(S)-APR	annual frequency, anchored end of April
(B)A(S)-MAY	annual frequency, anchored end of May
(B)A(S)-JUN	annual frequency, anchored end of June
(B)A(S)-JUL	annual frequency, anchored end of July
(B)A(S)-AUG	annual frequency, anchored end of August
(B)A(S)-SEP	annual frequency, anchored end of September
(B)A(S)-OCT	annual frequency, anchored end of October
(B)A(S)-NOV	annual frequency, anchored end of November
"""






