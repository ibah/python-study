# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:36:58 2017
Computational Statistics in Python
http://people.duke.edu/~ccc14/sta-663/index.html
(for exercises - see a separate file)

Content
Introduction
Algebra
"""

''' Introduction to Python
http://people.duke.edu/~ccc14/sta-663/IntroductionToPythonSolutions.html
'''

# typing is dynamic
a = "1"
print(type(a))
a = 1.0
print(type(a))
a = 1
print(type(a))
# flow control
for i in range(1,10):
     print(i)
i = 1
while i < 10:
    print(i)
    i+=1
a = 20
if a >= 22:
   print("if")
elif a >= 21:
    print("elif")
else:
    print("else")
a = "1"
try:
  b = a + 2
except:
  print(a, " is not a number")
# Strings
a = "A string of characters, with newline \n CAPITALS, etc."
print(a)
b=5.0
newstring = a + "\n We can format strings for printing %.2f"
print(newstring %b)
a = "ABC defg"
print(a[1:3])
print(a[0:5])
print(a.lower())
print(a.upper())
print(a.find('d'))
print(a.replace('de','a'))
print(a) # wasn't modified
a.count('B')
# lists
a_list = [1,2,3,"this is a string",5.3]
b_list = ["A","B","F","G","d","x","c",a_list,3]
print(b_list)
print(b_list[7:9])
a = [1,2,3,4,5,6,7]
a.insert(0,0)
print(a)
a.append(8)
print(a)
a.reverse()
print(a)
a.sort()
print(a)
a.pop()
print(a)
a.remove(3)
print(a)
a.remove(a[4])
print(a)
# list comprehensions
even_numbers = [x for x in range(100) if x % 2 == 0]
print(even_numbers)
first_sentence = "It was a dark and stormy night."
characters = [x for x in first_sentence]
print(characters)
def sqr(x): return x ** 2
a = [2,3,4]
b = [10,5,3]
c = map(sqr,a) # applies function to all elements of a lists; returns an iterator
print(list(c))
d = map(pow,a,b)
print([x for x in d])
# tuples
# unchangable lists + unpacking
my_pets = ("Chestnut", "Tibbs", "Dash", "Bast")
(aussie,b_collie,indoor_cat,outdoor_cat) = my_pets
print(aussie)
cats=(indoor_cat,outdoor_cat)
print(cats)
# dictionaries
# unordered, keyed lists
a = {'anItem': "A", 'anotherItem': "B",'athirdItem':"C",'afourthItem':"D"}
print(a)
print(a[1])
print(a['anItem'])
# sets
# unordered collections of unique elements
# support: intersections, unions and set differences
# can be used to remove duplicates from a collection or to test for membership
fruits = set(["apples","oranges","grapes","bananas"])
citrus = set(["lemons","oranges","limes","grapefruits","clementines"])
citrus_in_fruits = fruits & citrus   #intersection
print(citrus_in_fruits)
diff_fruits = fruits - citrus        # set difference
print(diff_fruits)
diff_fruits_reverse = citrus - fruits  # set difference
print(diff_fruits_reverse)
citrus_or_fruits = citrus | fruits     # set union
print(citrus_or_fruits)
a_list = ["a", "a","a", "b",1,2,3,"d",1]
print(a_list)
a_set = set(a_list)  # Convert list to set
print(a_set)         # Creates a set with unique elements
new_list = list(a_set) # Convert set to list
print(new_list)        # Obtain a list with unique elements
type(set())
type({})
# classes
# obj.elem
# modules
# updating


# Exercises
# 1
for i in range(1,101):
    if i%15==0:
        print('FizzBuzz')
    elif i%3==0:
        print('Fizz')
    elif i%5==0:
        print('Buzz')
    else:
        print(i)
# 2
x=3;y=4
y,x = x,y
print(x,y)
# 3
def EucDis(a, b):
    ax, ay = a
    bx, by = b
    return np.sqrt((bx-ax)**2+(by-ay)**2)
def EucDis(a, b):
    return ((b[0]-a[0])**2 + (b[1]-a[1])**2)**0.5
print(EucDis((3,0),(0,4)))
# 4

# <------------------------------------------------------------------------------

####################################################################################

''' Getting started with Python and the IPython notebook
http://people.duke.edu/~ccc14/sta-663/IPythonNotebookIntroduction.html
'''

# (see the jupyter notebook)

###################################################################################

''' Functions
http://people.duke.edu/~ccc14/sta-663/FunctionsSolutions.html
'''

# Functions are first class objects
def square(x):
    return x*x
def cube(x):
    return x*x*x
funcs = {
    'square': square,
    'cube': cube
}
x=2
print(square(x))
print(cube(x))
for func in sorted(funcs):
    print(func, funcs[func](x))

# arguments of f(x):
# if passing a mutable object (list, dictionary) as argument then:
#   x is a copy of the original name of the object
#   if x is modified, then the external object is modified
#   if x is reassinged (e.g. x=a) then the external object remains unchanged

# Binding of default arguments occurs at function definition
def f(x = []): # v.1
    x.append(1)
    return x
print(f()) # see the unexpected behaviour
print(f())
print(f())
print(f(x = [9,9,9]))
print(f())
print(f())
def f(x = None): # v.2
    if x is None:
        x = []
    x.append(1)
    return x
print(f())
print(f())
print(f())
print(f(x = [9,9,9]))
print(f())
print(f())

# higher order function
# function that uses another function as an input argument or returns a function

# function as an input

# map, filter -> iterators, use in loop function
#               alternative: list comprehension
# map:
list(map(square, range(5)))
[square(x) for x in range(5)]
[x*x for x in range(5)]
# filter:
def is_even(x):
    return x%2==0
list(filter(is_even, range(5)))
[x for x in range(5) if is_even(x)]
[x for x in range(5) if x%2==0]
list(map(square, filter(is_even, range(5))))
[x**2 for x in range(5) if x%2==0]
# reduce -> use a for loop instead
#       alternative: functools.reduce
def my_add(x,y):
    return x+y
import functools
a = [1,2,3,4,5]
functools.reduce(my_add, a)
s = 0 # loop version
for x in a:
    s+=x
print(s)
def custom_sum(xs, transform):
    """Returns the sum of xs after a user specified transform."""
    return sum(map(transform,xs))
xs = range(5)
print(custom_sum(xs, square))
print(custom_sum(xs, cube))

# function as an output

def make_logger(target):
    def logger(data):
        with open(target, 'a') as f:
            f.write(data+'\n')
    return logger
import os
tmp = os.getcwd()
os.chdir(r'D:\\data\\Dropbox\\cooperation\\_python')
os.getcwd()
foo_logger = make_logger('foo.txt')
foo_logger('Hello')
foo_logger('World')
!cat 'foo.txt'

# anonymous functions
list(map(lambda x: x*x, range(5))) # iterator, use list comprehension if you need a list

# pure functions
# do not have any side effects and
# do not depend on global variables
# -> easier for debug & parall proc.
# impure:
# e.g. changing the passed object inside of the function

# Note that mutable functions [and defaults for their arguments] are created upon function declaration, not use.
# This gives rise to a common source of beginner errors.
def f1(x, y=[]):
    """Never give an empty list or other mutable structure as a default."""
    y.append(x)
    return sum(y)
print(f1(10))
print(f1(10))
print(f1(10, y =[1,2]))
print(f1(10))
# Here is the correct Python idiom
def f2(x, y=None):
    """Check if y is None - if so make it a list."""
    if y is None:
        y = []
    y.append(x)
    return sum(y)
print(f2(10))
print(f2(10))
print(f2(10, y =[1,2]))
print(f2(10))

# Recursion
def fact(n):
    """Returns the factorial of n."""
    if n==0:
        return 1
    else:
        return n*fact(n-1)
print([fact(n) for n in range(10)])
# Note that the recursive version is much slower than the non-recursive version
# Fibonacci
def fib1(n):
    """Fib with recursion."""
    # base case
    if n==0 or n==1:
        return 1
    # recurssive caae
    else:
        return fib1(n-1) + fib1(n-2)
print([fib1(i) for i in range(10)])
def fib2(n):
    """Fib without recursion."""
    a, b = 0, 1
    for i in range(1, n+1):
        a, b = b, a+b
    return b
print([fib2(i) for i in range(10)])
%timeit fib1(20)
%timeit fib2(20)

# Iterators
#xs = [1,2,3]
#x_iter = iter(xs)
#x_iter.next()
# Generators
#def count_down(n):
#    for i in range(n, 0, -1):
#        yield i
#counter = count_down(10)
#print(counter)
#print(counter.next())
#print(counter.next())
#for count in counter:
#    print(count)
xs1 = [x*x for x in range(5)]
print(xs1)
xs2 = (x*x for x in range(5))
print(xs2)

# A generatorr expression
print((x for x in range(10)))
# A list comprehesnnion
print([x for x in range(10)])
# A set comprehension
print({x for x in range(10)})
# A dictionary comprehension
print({x: x for x in range(10)})

# Decorators
# take a function and return a wrapped function that provides additional useful properties

# Here is a simple decorator to time an arbitrary function
def func_timer(func):
    """Times how long the function took."""
    def f(*args, **kwargs):
        import time
        start = time.time()
        results = func(*args, **kwargs)
        print("Elapsed: %.2fs" % (time.time() - start))
        return results
    return f

# There is a special shorthand notation for decorating functions
@func_timer
def sleepy(msg, sleep=1.0):
    """Delays a while before answering."""
    import time
    time.sleep(sleep)
    print(msg)
sleepy("Hello", 1.5)

# The operator module
# provides “function” versions of common Python operators (+, *, [] etc)
# that can be easily used where a function argument is expected

import operator as op
import functools as ft
# Here is another way to express the sum function
ft.reduce(op.add, range(10))
# The pattern can be generalized
ft.reduce(op.mul, range(1, 10))
# itemgetter []
my_list = [('a', 1), ('bb', 4), ('ccc', 2), ('dddd', 3)]
# standard sort
sorted(my_list)
# return list sorted by element at position 1 (remember Python counts from 0)
sorted(my_list, key=op.itemgetter(1))
# the key argument is quite flexible
sorted(my_list, key=lambda x: len(x[0]), reverse=True)

# The functools module
from functools import partial
sum_ = partial(ft.reduce, op.add)
prod_ = partial(ft.reduce, op.mul)
sum_([1,2,3,4])
prod_([1,2,3,4])

# This is extremely useful to create functions
# that expect a fixed number of arguments
import scipy.stats as stats
def compare(x, y, func):
    """Returne p-value for some appropriate comparison test."""
    return func(x, y)[1]
x, y = np.random.normal(0, 1, (100,2)).T
print("p value assuming equal variance    =%.8f" % compare(x, y, stats.ttest_ind))
test = partial(stats.ttest_ind, equal_var=False)
print("p value not assuming equal variance=%.8f" % compare(x, y, test))

# The itertools module
# provides many essential functions for working with iterators
#   permuations and combinations generators may be particularly useful for simulations
#   groupby generator is useful for data analysis.
from itertools import cycle, groupby, islice, permutations, combinations
# cycle, islice
list(islice(cycle('abcd'), 0, 10))
# groupby
animals = sorted(['pig', 'cow', 'giraffe', 'elephant',
                  'dog', 'cat', 'hippo', 'lion', 'tiger'], key=len)
for k, g in groupby(animals, key=len):
    print(k, list(g))
# permutation
[''.join(p) for p in permutations('abc')]
[list(p) for p in permutations('abc')]
[p for p in permutations('abc')]
for p in permutations('abc'):
    print(p)
# combination
[list(c) for c in combinations([1,2,3,4], r=2)]
[c for c in combinations([1,2,3,4], r=2)]
for c in combinations([1,2,3,4], r=2):
    print(c)

# functional style
# modules: toolz, fn, funcy



# Exercises



# 1
ans = []
for i in range(3):
    for j in range(4):
        ans.append((i, j))
ans
[(i,j) for i in range(3) for j in range(4)]
# 2

# <--------------------------------------------------------------------------------------------




















'''Obtaining Data '''
# Remote data
# Plain text files
#import urllib2
#text = urllib2.urlopen('http://www.gutenberg.org/cache/epub/11/pg11.txt').read()
import requests
test = requests.get('http://www.gutenberg.org/cache/epub/11/pg11.txt').text

# <--------------------------------------------------------------------------------------------


''' Working with text
http://people.duke.edu/~ccc14/sta-663/TextProcessingSolutions.html
'''

# Exercises

###############################################################################

''' Using Numpy
http://people.duke.edu/~ccc14/sta-663/UsingNumpySolutions.html
'''

import numpy as np
# Create a 10 by 6 array from normal deviates and convert to ints
n, nrows, ncols = 100, 10, 6
xs = np.random.normal(n, 15, size=(nrows, ncols)).astype('int')
xs
# indexing: slices, lists of indeces, boolean
a = np.arange(16).reshape(4,4)
a
np.tril(a)
np.tril(a, -1)
np.diag(a)
np.diag(np.diag(a))
np.triu(a)
np.triu(a, 1)
# broadcasting
xs = np.arange(12).reshape(2,6)
xs
# broadcasting just works when doing column-wise operations
col_means = xs.mean(axis=0)
col_means
xs - col_means
# but needs a little more work for row-wise operations
row_means = xs.mean(axis=1)[:, np.newaxis]
row_means
xs - row_means
# convert matrix to have zero mean and unit standard deviation using col summary statistics
(xs - xs.mean(axis=0))/xs.std(axis=0)
# convert matrix to have zero mean and unit standard deviation using row summary statistics
(xs - xs.mean(axis=1)[:, np.newaxis])/xs.std(axis=1)[:, np.newaxis]
# broadcasting for outer product
# e.g. create the 12x12 multiplication toable
u = np.arange(1, 13)
u[:,None] * u[None,:]
np.outer(u,u)
np.einsum('i,j->ij',u,u)
# Calculate the pairwise distance matrix between the following points
pts = np.array([(0,0), (4,0), (4,3), (0,3)])
# loop solution
def distance_matrix_py1(pts):
    """Returns matrix of pairwise Euclidean distances. Pure Python version.
    works for 2-D space only"""
    n = len(pts)
    m = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            m[i,j] = ((pts[i,0]-pts[j,0])**2 + (pts[i,1]-pts[j,1])**2)**0.5
    return m
distance_matrix_py1(pts)
def distance_matrix_py2(pts):
    """Returns matrix of pairwise Euclidean distances. Pure Python version.
    works for any number of dimensions of the space"""
    n = len(pts) # number of points
    p = len(pts[0]) # number of dimensions
    m = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(p):
                s += (pts[i,k]-pts[j,k])**2
            m[i,j] = s**0.5
    return m
distance_matrix_py2(pts)
# numpy version
n, p = pts.shape # fater retrial of n and p
def distance_matrix_np(pts):
    return ((pts[None,:] - pts[:,None])**2).sum(-1)**0.5
distance_matrix_np(pts)
# checking
pts - pts
pts[None,:]
pts[:,None]
pts[None,:] - pts[:,None]
(pts[None,:] - pts[:,None])**2
np.sum((pts[None,:] - pts[:,None])**2, -1)
((pts[None,:] - pts[:,None])**2).sum(-1)
((pts[None,:] - pts[:,None])**2).sum(-1)**0.5
# Broaccasting and vectorization is faster than looping
%timeit distance_matrix_py1(pts)
%timeit distance_matrix_py2(pts)
%timeit distance_matrix_np(pts)

# Universal functions
# work both on scalars and arrays (element-wise)
xs = np.linspace(0, 2*np.pi, 100)
ys = np.sin(xs) # np.sin is a universal function
import matplotlib.pyplot as plt
plt.plot(xs, ys)
# operators also perform elementwise operations by default
xs = np.arange(10)
print(xs)
print(-xs)
print(xs+xs)
print(xs*xs)
print(xs**3)
print(xs < 5)

# Generalized Functions
#  performs looping over vectors or arrays
from numpy.core.umath_tests import matrix_multiply
print(matrix_multiply.signature)
us = np.random.random((5, 2, 3)) # 5 2x3 matrics
vs = np.random.random((5, 3, 4)) # 5 3x4 matrices
# perform matrix multiplication for each of the 5 sets of matrices
ws = matrix_multiply(us, vs)
print(ws.shape)
print(ws)
np.einsum('imn,inp->imp',us,vs) # the same usign einsum

# Random numbers
# numpy.random: uniform, mutinomial, shuffle, permutation, choice
# scipy.stats: norm
#   frozen dist: rvs, pdf, cdf, ppf

import numpy.random as npr
roll = 1/6
# 100 trials of a die being rolled 10 times
x = npr.multinomial(100, [roll]*6, 10)
x
# Throw a die 20 times:
np.random.multinomial(20, [1/6.]*6, size=1)
# uniformly distributed numbers in 2D
x = npr.uniform(-1, 1, (100, 2))
x
plt.scatter(x[:,0], x[:,1], s=50); plt.axis([-1.05, 1.05, -1.05, 1.05]);
# ranodmly shuffling a vector
x = np.arange(10)
npr.shuffle(x)
x
# radnom permutations
npr.permutation(10)
# radnom selection without replacement
x = np.arange(10,20)
npr.choice(x, 10, replace=False)
# radnom selection with replacement
npr.choice(x, (5, 10), replace=True) # this is default
# toy example - estimating pi inefficiently
n = 1e6
x = npr.uniform(-1,1,(n,2))
4*(x[:,0]**2 + x[:,1]**2 < 1).sum()/n

import scipy.stats as stats
# Create a "frozen" distribution - i.e. a partially applied function
dist = stats.norm(10, 2)
dist.rvs(10) # rnorm
dist.pdf(np.linspace(5,15,10)) # pnorm
dist.cdf(np.linspace(5,15,11)) # dnorm
dist.ppf(dist.cdf(np.linspace(5,15,11))) # qnorm

''' Linear Algebra '''

# the linear algebra functions can be found in scipy.linalg
import numpy as np
import scipy.linalg as la
from functools import reduce
from numpy import random as npr

# Matrix operations
A = np.array([[1,2],[3,4]])
b = np.array([1,4])
print(A)
print(b)
print(np.dot(A, A))
print(A)
print(la.inv(A))
print(A.T)
x = la.solve(A, b) # do not use x = dot(inv(A), b) as it is inefficient and numerically unstable
print(x)
print(np.dot(A, x) - b)

# Matrix decompositions
x = npr.normal(100, 15, (6,10))
A = np.floor(x)
print(A)
# pivoted LU decomposition of a matrix
P, L, U = la.lu(A)
print(np.dot(P.T, A))
print(np.dot(L, U))
# QR decomposition of a matrix
# Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal
and R upper triangular
Q, R = la.qr(A)
print(A)
print(np.dot(Q, R))
# Singular Value Decomposition
# Factorizes the matrix a into two unitary matrices U and Vh, and
# a 1-D array s of singular values (real, non-negative) such that
# ``a == U*S*Vh``, where S is a suitably shaped matrix of zeros with main diagonal s.
U, s, V = la.svd(A)
m, n = A.shape
S = np.zeros((m,n))
for i, _s in enumerate(s):
    S[i,i] = _s
print((reduce(np.dot, [U, S, V])))
# Covariance
B = np.cov(A)
print(B)
# Solve an ordinary or generalized eigenvalue problem of a square matrix.
# Find eigenvalues w and right or left eigenvectors of a general matrix
u, V = la.eig(B)
print(np.dot(B, V))
print(np.real(np.dot(V, np.diag(u))))
# Cholesky decomposition
C = la.cholesky(B)
print(np.dot(C.T, C))
print(B)

# Covariance Matrix
np.random.seed(123)
x = npr.multivariate_normal([10,10], np.array([[3,1],[1,5]]), 10)
# create a zero mean array
u = x - x.mean(0)
cov = np.dot(u.T, u)/(10-1)
print(cov)
print(np.cov(x.T))

# Least squares solution
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.linalg as la

# Set up a system of 11 linear equations
x = np.linspace(1,2,11)
y = 6*x - 2 + npr.normal(0, 0.3, len(x))
plt.plot(x, y, 'o')
# Form the VanderMonde matrix
A = np.vstack([x, np.ones(len(x))]).T
A
# The linear algebra librayr has a lstsq() function
# that will do the above calculaitons for us
b, resids, rank, sv = la.lstsq(A, y)
b # original (6, -2)
resids
rank
sv
# Check against pseudoinverse and the normal equation
print("lstsq solution".ljust(30), b)
print("pseudoinverse solution".ljust(30), np.dot(la.pinv(A), y))
print("normal euqation solution".ljust(30), np.dot(np.dot(la.inv(np.dot(A.T, A)), A.T), y))
# Now plot the solution
xi = np.linspace(1,2,11)
yi = b[0]*xi + b[1]
plt.plot(x, y, 'o');plt.plot(xi, yi, 'r-');

# Least squares for polynomials
x = np.linspace(0,2,11)
y = 6*x*x + .5*x + 2 + npr.normal(0, 0.6, len(x))
plt.plot(x, y, 'o')
A = np.vstack([x*x, x, np.ones(len(x))]).T
A
b = la.lstsq(A, y)[0]
b
xi = np.linspace(0,2,11)
yi = b[0]*xi*xi + b[1]*xi + b[2]
plt.plot(xi, yi, 'r-'); plt.plot(x, y, 'o')

# easier OLS fit for polynomials
b = np.random.randint(0, 10, 6)
x = np.linspace(0, 1, 25)
y = np.poly1d(b)(x) # A one-dimensional polynomial class.
y += np.random.normal(0, 5, y.shape)
b
x
y
p = np.poly1d(np.polyfit(x, y, len(b)-1)) # Least squares polynomial fit (x, y, degrees of freedom)
print(p)
plt.plot(x, y, 'bo'); plt.plot(x, p(x), 'r-')
list(zip(b, p.coeffs))

# Exercises

#####################################################################################

''' Using Pandas
http://people.duke.edu/~ccc14/sta-663/UsingPandas.html
'''
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas import Series, DataFrame, Panel
from string import ascii_lowercase as letters
from scipy.stats import chisqprob
import matplotlib.pyplot as plt

# Series
# a 1D array with axis labels
letters
list(letters[:10])
tuple(letters[:10])
xs = Series(np.arange(10), index=tuple(letters[:10]))
xs
xs[:3]
xs[::3]
xs[['d','f','h']]
xs['b':'d']
xs.a
# All the numpy functions wiill work with Series objects, and return another Series
y1, y2 = np.mean(xs), np.var(xs)
y1, y2
# Matplotlib will work on Series objects too
plt.plot(xs, np.sin(xs))
plt.plot(xs, np.sin(xs), 'r-o')
plt.plot(xs, np.sin(xs), 'r-o', xs, np.cos(xs))
plt.plot(xs, np.sin(xs), 'r-o', xs, np.cos(xs), 'b-x')
# Convert to numpy arrays with values
xs.values

# time series
import datetime as dt
from pandas import date_range
print(dt.date.today())
today = dt.datetime.strptime('Jan 21 2015', '%b %d %Y')
print(today)
days = date_range(today, periods=35, freq='D')
print(days)
ts = Series(np.random.normal(10, 1, len(days)), index=days)
print(ts.head())
# Extracting elements
print(ts[0:4])
print(ts['2015-01-21':'2015-01-28'])# Note - includes end time
# We can geenerate statistics for time ranges with the resample method
# For example, suppose we are interested in weekly means, standard deviations and sum-of-squares
df = ts.resample(rule='W', how=('mean', 'std', lambda x: sum(x*x)))
df # weekly resample

# data frame
type(df)
# renaming columns
df.columns = ('mu', 'sigma', 'sum_of_sq')
df
# Extracitng columns from a DataFrame
df.mu # by attribute
df['sigma'] # by column name
# Extracting rows from a DataFrame
df[1:3]
df['2015-01-21'::2]
# Extracting blocks and scalars
df.iat[2, 2] # extract an element with iat() (by integer position)
df.loc['2015-01-25':'2015-03-01', 'sum_of_sq'] # indexing by label
df.iloc[:3, 2] # indexing by position
df.ix[:3, 'sum_of_sq'] # by label OR position
# Using Boolean conditions for selecting eleements
df[(df.sigma < 1) & (df.sum_of_sq < 700)] # (faster) need parenthesis because of operator precedence
df.query('sigma < 1 and sum_of_sq < 700') # (slower) the query() method allows more readable query strings

# Panels
# 3D arrays
# dictionaries of DataFrames
import pandas as pd
df= np.random.binomial(100, 0.95, (9,2))
dm = np.random.binomial(100, 0.9, [12,2])
df
dm
dff = pd.DataFrame(df, columns = ['Physics', 'Math'])
dfm = pd.DataFrame(dm, columns = ['Physics', 'Math'])
dff
dfm
score_panel = pd.Panel({'Girls': dff, 'Boys': dfm})
score_panel
score_panel.Girls.transpose()
# find physics and math scores of girls who scored >= 93 in math
score_panel.Girls[score_panel.Girls.Math >= 93] # DataFrame is returned
score_panel.ix['Girls',score_panel.Girls.Math >= 93]

# Split-Apply-Combine
# http://people.duke.edu/~ccc14/sta-663/UsingPandas.html#split-apply-combine
# import a DataFrame to play with
try:
    tips = pd.read_pickle('tips.pic')
except:
    tips = pd.read_csv('https://raw.github.com/vincentarelbundock/Rdatasets/master/csv/reshape2/tips.csv', )
    tips.to_pickle('tips.pic')
tips.head()
# get read of the first column
# (you could do that nicely in pd.read_csv)
tips = tips.ix[:, 1:]
tips.head()
# counts by sex and smoker status
tips.columns
tips.groupby(['sex','smoker']).count()
tips.groupby(['sex','smoker']).size() # correct
# to get margins:
pd.crosstab(tips.sex, tips.smoker)
pd.crosstab(tips.sex, tips.smoker, margins=True)
tips.pivot_table(values='tip', index='sex', columns='smoker', aggfunc='count', margins=True).astype(int)
#   values is set only to drop all other counts
#   astype int is needed as after adding margins the values were changed to floats
# If more than 1 column of resutls is generated, a DataFrame is returned
grouped = tips.groupby(['sex','smoker'])
grouped.size() # Series
grouped.mean() # DataFrame
# The returned results can be further manipulated via apply()
# For example, suppose the bill and tips are in USD but we want EUR
import json
import urllib
# get current conversion rate  <- doesn't work
#converter = json.loads(urllib.urlopen('http://rate-exchange.appspot.com/currency?from=USD&to=EUR').read())
#urllib.request.URLopener.open(fullurl='http://rate-exchange.appspot.com/currency?from=USD&to=EUR')
#print(converter)
converter = {u'to': u'EUR', u'rate': 0.879191, u'from': u'USD'}
grouped['total_bill', 'tip'].mean().apply(lambda x: x*converter['rate'])
rate = 2
grouped['total_bill', 'tip'].mean().apply(lambda x: x*rate)
# We can also transform the original data for more convenient analysis
# For example, suppose we want standardized units for total bill and tips
zscore = lambda x: (x - x.mean())/x.std()
std_grouped = grouped['total_bill', 'tip'].transform(zscore)
std_grouped.head(n=4)
# Suppose we want to apply a set of functions to only some columns
grouped['total_bill', 'tip'].agg(['mean', 'min', 'max'])
# We can also apply specific functions to specific columns
df = grouped.agg({'total_bill': (min, max), 'tip': sum})
df

''' Using statsmodels
http://people.duke.edu/~ccc14/sta-663/UsingPandas.html#using-statsmodels
'''
# statsmodels package:
# replicates many of the basic statistical tools available in R

# Simulate the genotype for 4 SNPs in a case-control study using an additive genetic model
n = 1000
status = np.random.choice([0,1], n )
genotype = np.random.choice([0,1,2], (n,4))
genotype[status==0] = np.random.choice([0,1,2], (sum(status==0), 4), p=[0.33, 0.33, 0.34])
genotype[status==1] = np.random.choice([0,1,2], (sum(status==1), 4), p=[0.2, 0.3, 0.5])
df = pd.DataFrame(np.hstack([status[:, np.newaxis], genotype]), columns=['status', 'SNP1', 'SNP2', 'SNP3', 'SNP4'])
df.head(6)
# Use statsmodels to fit a logistic regression to  the data
import statsmodels.api as sm
# formula:
formula = 'status ~ %s' % '+'.join(df.columns[1:]) # 'status ~ SNP1+SNP2+SNP3+SNP4'
fit1 = sm.Logit.from_formula(formula , data=df).fit()
fit1.summary()
# # Alternative using GLM - similar to R
fit2 = sm.GLM.from_formula(formula, data=df, family=sm.families.Binomial()).fit()
fit2.summary() # the same output
# (central) chi-squared distribution:
# goodness of fit for qualitative data
# np.chisqprob(fit2.null_deviance - fit2.deviance, fit2.df_model) # deprecated
from scipy.stats import chi2
chi2.sf(fit2.null_deviance - fit2.deviance, fit2.df_model) # survival function
#   so the statistic is very high, far in the right tail of the distribution
(fit2.null_deviance - fit2.deviance, fit2.df_model)




"""
Computational problems in statistics
http://people.duke.edu/~ccc14/sta-663/ComputationalStatisticsMotivation.html#computational-problems-in-statistics
"""



import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline
%precision 4
np.random.seed(1)
plt.style.use('ggplot')

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# e.g. is coin fair?

# Hypothesis Test
# How well does the data match some assumed (null) distribution?
n = 100
pcoin = 0.62 # actual value of p for coin
results = st.bernoulli(pcoin).rvs(n)
h = sum(results)
h
# Expected distribution for fair coin
p = 0.5
rv = st.binom(n, p)
mu = rv.mean()
sd = rv.std()
mu, sd
# using binomial test
st.binom_test(h, n, p) # very unlikely result
# Using z-test approximation with continuity correction
z = (h-mu)/sd # original code: h-0.5-mu (I don't know why)
z # very high z-score
2*(1-st.norm.cdf(z)) # 2x upper tail
2*st.norm.sf(z) # 2x upper tail as survival function
# Using simulation to estimate null distribution
nsamples = 100000
# numpy:
xs = np.random.binomial(n, p, nsamples) # numpy
2*np.sum(xs >= h)/xs.size
# scipy.stats
xs2 = st.binom(n,p).rvs(nsamples)
2*np.sum(xs2 >= h)/xs2.size

# Point Estimate
# If it doesn’t match well but we think it is likely to belong to
# a known family of distributions, can we estiamte the parameters?
# Maximum likelihood estimate of pcoin
print("Maximum likelihood", np.sum(results)/len(results))

# Interval Estimates
# How accurate are the parameter estimates?
# Using bootstrap to esitmate confidenc intervals for pcoin
bs_samples = np.random.choice(results, (nsamples, len(results)), replace=True)
bs_samples.shape
bs_ps = np.mean(bs_samples, axis=1)
bs_ps.shape # means for the 100,000 bootstrapped samples
bs_ps.sort()
bs_ps[int(0.025*nsamples)] # left end of the 95% interval
bs_ps[int(0.975*nsamples)] # right end of the 95% interval
print("Bootstrap CI: (%.4f, %.4f)" % (bs_ps[int(0.025*nsamples)], bs_ps[int(0.975*nsamples)]))
# Using binomial quantiles
print("Binomial CI: (%.4f, %.4f)" % (st.binom(n, phat).ppf(0.025)/n, st.binom(n, phat).ppf(0.975)/n))
# Using normal approximation
phat = np.sum(results)/n
phat # estimate of p
sd1 = np.std(results,ddof=1)
sd1 # std.dev taken from the sample - use this one
sd2 = np.sqrt(phat*(1-phat))
sd2 # std.dev taken from the assumption on the distribution shape / family
print("CLT CI: (%.4f, %.4f)" % tuple(phat + np.array([-1,1])*st.norm().ppf(0.975) * sd1 / np.sqrt(n)))


# function estimation or approximation
# Can we estimate the entire distribution?
# Bayesian Approach:
# directly estimates the posterior distribution,
# from which all other point/interval statistics can be estimated.
a, b = 10, 10
prior = st.beta(a, b)
post = st.beta(h+a, n-h+b)
ci = post.interval(0.95)
map_ =(h+a-1.0)/(n+a+b-2.0)
prior, post
prior.interval(0.95), ci
map_

xs = np.linspace(0, 1, 100)
plt.plot(prior.pdf(xs), label='Prior')
plt.plot(post.pdf(xs), label='Posterior')
plt.axvline(mu, c='red', linestyle='dashed', alpha=0.4) # 0.5
plt.xlim([0, 100])
plt.axhline(0.3, ci[0], ci[1], c='black', linewidth=2, label='95% CI');
plt.axvline(n*map_, c='blue', linestyle='dashed', alpha=0.4)
plt.legend();

''' Comment
All the above calculations have simple analytic solutions.
For most real life problems reuqireing more complex statistical models,
we will need to search for solutions using more advanced numerical methods and simulations.
However, the types of problems that we will be addressing are largely similar
to those asked of the toy coin toss problem. These include

- point estimation (e.g. summary statistics)
- interval estimation (e.g. confidence intervals or Bayesian credible intervals)
- function estimation (e.g. density estimation, posteriro distributions)

and most will require some knowledge of numerical methods for

- optimization (e.g. least squares minimizaiton, maximum likelihood)
- Monte Carlo simulations (e.g. Monte Carlo integration, MCMC, bootstrap, permutation-resampling)

The next section of the course will focus on the ideas behiind these numerical methods.
'''

"""
Computer numbers and mathematics
http://people.duke.edu/~ccc14/sta-663/ComputerArithmetic.html#computer-numbers-and-mathematics
"""
def plot_func(func, x_range=(-1,1)):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = func(x)
    plt.plot(x, y)
    plt.show()

# Examples of issues

# 1 - this is OK
def normalize(ws):
    """Returns normalized set of weights that sum to 1."""
    s = sum(ws)
    return [w/s for w in ws]
ws = [1,2,3,4,5]
normalize(ws) # this is OK
# Fix - not needed.

# 2 - this is some issue
# multiplying many small numbers -> result is 0
from scipy.stats import norm
rv1 = norm(0, 1)
rv2 = norm(0, 3)
plot_func(rv1.pdf)
plot_func(rv2.pdf)
plt.plot(xs, rv1.pdf(xs), 'bo', xs, rv2.pdf(xs), 'ro')#; plt.xlim([-1,2])
xs = np.random.normal(0, 3, 1000)
xs.shape
likelihood1 = np.prod(rv1.pdf(xs)) # strange, the product is 0, while there are no 0's (all elements > 0)
likelihood2 = np.prod(rv2.pdf(xs)) # ditto
likelihood2 > likelihood1
# Fix
# Work in log space for very small or very big numbers to reduce underflow/overflow
probs = np.random.random(1000)
np.prod(probs) # 0 (wrong)
np.sum(np.log(probs)) # should be the same
np.exp(np.sum(np.log(probs))) # this is 0 again, this number is too low


# 3 - the basic example of 0.1 not having an exact representation
s = 0.0
for i in range(1000):
    s += 1.0/10.0
    if s == 1.0:
        break
print(i)
# Fix
TOL = 1e-9
s = 0.0
for i in range(1000):
    s += 1.0/10.0
    if abs(s - 1.0) < TOL:
        break
print(i)

# 4
def var(xs):
    """Returns variance of sample data."""
    n = 0
    s = 0
    ss = 0
    xs = xs - np.mean(xs) ### here I added it
    # as the formula for v makes sense only if mean(x) is 0
    for x in xs:
        n +=1
        s += x
        ss += x*x
    v = (ss - (s*s)/n)/(n-1)
    return v
# What is the sample variance for numbers from a normal distribution with variance 1?
np.random.seed(4)
xs = np.random.normal(1e9, 1, 1000)
var(xs)

# Associative law
6.022e23 - 6.022e23 + 1 # 1
1 + 6.022e23 - 6.022e23 # 0

# distributive law
a = np.exp(1);
b = np.pi;
c = np.sin(1);
a*(b+c) == a*b+a*c # False

# limits
import sys
# integers
sys.maxsize
2**63-1 == sys.maxsize
sys.maxsize + 1 # # swithing from integers to "long" abritrary precsion numbers
type(sys.maxsize + 1) # it says int, but internal representation might be different?
# floats
sys.float_info

# Continue....
# http://people.duke.edu/~ccc14/sta-663/ComputerArithmetic.html#overflow-in-langauges-such-as-c-wraps-around-and-gives-negative-numbers

# <--------------------------------------------------


# arbitrary precision libraries
# pip install gmpy2
# pip install mpmath
# http://people.duke.edu/~ccc14/sta-663/ComputerArithmetic.html#using-arbitrary-precision-libraries


"""
Algorithmic complexity
http://people.duke.edu/~ccc14/sta-663/AlgorithmicComplexity.html
"""
# Use %timeit to measure function calls
def f():
    import time
    time.sleep(2)
%timeit -n1 f()

# Measuring algorithmic complexity
def f1(n, k):
    return k*n*n
def f2(n, k):
    return k*n*np.log(n)
n = np.arange(0, 20001)
plt.plot(n, f1(n, 1), c='blue')
plt.plot(n, f2(n, 1000), c='red')
plt.xlabel('Size of input (n)', fontsize=16)
plt.ylabel('Number of operations', fontsize=16)
plt.legend(['$\mathcal{O}(n^2)$', '$\mathcal{O}(n \log n)$'], loc='best', fontsize=20);

# Searching a list is O(n)
alist = range(1000000)
r = np.random.randint(100000)
%timeit -n3 r in alist
# Searching a dictionary is O(1)
adict = dict.fromkeys(alist) # just keys, the values are all None
%timeit -n3 r in adict

# Space complexity
# Notice how much overhead Python objects have
# A raw integer should be 64 bits or 8 bytes only
sys.getsizeof(1)
sys.getsizeof(1234567890123456789012345678901234567890)
sys.getsizeof(3.14)
sys.getsizeof(3j)
sys.getsizeof('a')
sys.getsizeof('hello world')
np.ones((100,100), dtype='byte').nbytes
np.ones((100,100), dtype='i2').nbytes
np.ones((100,100), dtype='int').nbytes # default is 64 bits or 8 bytes
np.ones((100,100), dtype='f4').nbytes
np.ones((100,100), dtype='float').nbytes # default is 64 bits or 8 bytes
np.ones((100,100), dtype='complex').nbytes

"""
Linear Agebra and Linear Systems
http://people.duke.edu/~ccc14/sta-663/LinearAlgebraReview.html#linear-algebra-and-linear-systems
"""
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%matplotlib inline
#%precision 4
plt.style.use('ggplot')

import numpy as np
from scipy import linalg
# norm of a vector = it's length = ||v||
v = np.array([1,2])
linalg.norm(v)
(v[0]**2+v[1]**2)**0.5 # ditto
# distance between two vectors
w = np.array([1,1])
linalg.norm(v-w)
# inner product = <v, w>
v.dot(w)
v.dot(v)**0.5 # = ||v||
#np.outer(v.T,w)
# outer product = v O w.T
np.outer(v,w)

# the covariance matrix is an outer product
n, p = 10, 4
v = np.random.random((p,n))
v.shape # p rows, n columns
np.cov(v) # 4x4
# From the definition, the covariance matrix
# is just the outer product of the normalized
# matrix where every variable has zero mean
# divided by the number of degrees of freedom
v.mean(1) # mean for every row
v-v.mean(1) # could not broadcast (4,10), (4,)
v.mean(1)[:, np.newaxis] # transposed into (4,1) 
w = v - v.mean(1)[:, np.newaxis]
w.mean(1) # 4x zero
w.dot(w.T) /  (n-1) # = cov(v), this is an inner product
np.cov(v)

# Trace and Determinant of Matrices
""" Trace
 It is an invariant of a matrix under change of basis (more on this later).
 It defines a matrix norm (more on that later)
 Determinant
 Like the trace, it is also invariant under change of basis
 An n×n matrix A is invertible ⟺ det(A)≠0
 The rows(columns) of an n×nn×n matrix A are linearly independent ⟺ det(A)≠0
 """
n = 6
M = np.random.randint(100,size=(n,n))
print(M)
np.trace(M)
np.linalg.det(M)

"""
Column space, Row space, Rank and Kernel

A(m*n)
column space = space of all linear combinations of a<n>
row space = ... a<m>
Dimensions of these spaces are equal for any matrix.
rank = dimension of the image of the linear transformation determined by A
kernel = dimension of the space mapped to zero under the linear transformation that A represents
nullity = dimension of the kernel of a linear transformation
Index theorem: For an m×nn matrix A, rank(A) + nullity(A) = n
"""

"""
Matrices as Linear Transformations

Multiplying vector by a matrix
= either rotate, reflect, dilate or some combination of those three
= linear transformation
Any matrix defines a linear transformation
The matrix form of a linear transformation is NOT unique
We need only define a transformation by saying what it does to a basis

Change of basis
BAB^-1, for any invertible matrix B

PCA = express the matrix in a basis of eigenvectors
"""
A = np.array([[2,1],[3,1]])  # transformation f in standard basis
e1 = np.array([1,0])         # standard basis vectors e1,e2
e2 = np.array([0,1])

print(A.dot(e1))             # demonstrate that Ae1 is (2,3)
print(A.dot(e2))             # demonstrate that Ae2 is (1,1)

# new basis vectors
v1 = np.array([1,3])
v2 = np.array([4,1])

# How v1 and v2 are transformed by A
print("Av1: ")
print(A.dot(v1))
print("Av2: ")
print(A.dot(v2))

# Change of basis from standard e1,e2 to v1,v2
B = np.array([[1,4],[3,1]])
print(B)
B.dot(e1) # = v1
B.dot(e2) # = v2
# so B transforms base (e1,e2) into base (v1,v2)
B_inv = np.linalg.inv(B)

print("B B_inv ")
print(B.dot(B_inv))   # check inverse

# transform e1 under change of coordinates
T = B.dot(A.dot(B_inv))        # B A B^{-1}
T
# = the transformation A translated from base e1,e2 into base v1,v2
T.dot(e1)
T.dot(v1)

coeffs = T.dot(e1)
coeffs

print(coeffs[0]*v1 + coeffs[1]*v2)
# I don't understand this...

def plot_vectors(vs):
    """Plot vectors in vs assuming origin at (0,0)."""
    n = len(vs)
    X, Y = np.zeros((n, 2))
    U, V = np.vstack(vs).T
    plt.quiver(X, Y, U, V, range(n), angles='xy', scale_units='xy', scale=1)
    xmin, xmax = np.min([U, X]), np.max([U, X])
    ymin, ymax = np.min([V, Y]), np.max([V, Y])
    xrng = xmax - xmin
    yrng = ymax - ymin
    xmin -= 0.05*xrng
    xmax += 0.05*xrng
    ymin -= 0.05*yrng
    ymax += 0.05*yrng
    plt.axis([xmin, xmax, ymin, ymax])
#e1 = np.array([1,0])
#e2 = np.array([0,1])
#A = np.array([[2,1],[3,1]])
# Here is a simple plot showing Ae_1 and Ae_2
# You can show other transofrmations if you like
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plot_vectors([e1, e2])
plt.subplot(1,2,2)
plot_vectors([A.dot(e1), A.dot(e2)])
plt.tight_layout()

plot_vectors([v1,v2])
# checking
vs = [v1,v2]
vs
n = len(vs)
X, Y = np.zeros((n, 2))
X, Y
U, V = np.vstack(vs).T
np.vstack(vs)
np.vstack(vs).T
U, V
plt.quiver(U, V)
plt.quiver(X, Y, U, V)
plt.quiver(X, Y, U, V, range(n))
plt.quiver(X, Y, U, V, range(n), angles='xy')
plt.quiver(X, Y, U, V, range(n), angles='xy', scale_units='xy')
# -> angles:
#    'xy': the arrow points from (x,y) to (x+u, y+v)
plt.quiver(X, Y, U, V, range(n), angles='xy', scale_units='xy', scale=1)
xmin, xmax = np.min([U, X]), np.max([U, X])
ymin, ymax = np.min([V, Y]), np.max([V, Y])
U, X
np.min([U, X])
xrng = xmax - xmin
yrng = ymax - ymin
xmin -= 0.05*xrng
xmax += 0.05*xrng
ymin -= 0.05*yrng
ymax += 0.05*yrng
plt.axis([xmin, xmax, ymin, ymax])



# Matrix Norms
# notion extended from vectors
# used in determining the condition of a matrix
# see more in the text

"""
Special Matrices
allow us either simplify the underlying linear system or to understand more about it.

matrices
  square
  diagonal
  (skew) symmetric: aij = (-)aji
  upper/lower triangular
  orthogonal: A * A^T = I, A^T = A^-1
    The rows and columns of an orthogonal matrix are an orthonormal set of vectors.
    Geometrically speaking, orthogonal transformations preserve lengths and angles between vectors
  positive definite:  u^T A u >0, u non-zero n-dimensional vector
    Symmetric, positive-definite matrices have ‘square-roots’ (in a sense)
    Any symmetric, positive-definite matrix is diagonizable!!!
    Co-variance matrices are symmetric and positive-definite
  diagonalizable
    exists an inverible matrix P, P^-1 * A * P is a diagonal matrix
    A square matrix that is not diagonalizable is called defective
    especially easy to handle
      their eigenvalues and eigenvectors are known
      one can raise A to a power by simply raising the diagonal entries to that same power
      det(A) is simply the product of all entries
    geometrically, a diagonalizable matrix is an inhomogeneous dilation (or anisotropic scaling) — it scales the space, as does a homogeneous dilation, but by a different factor in each direction, determined by the scale factors on each axis (diagonal entries).
"""

"""
Linear Algebra and Matrix Decompositions
http://people.duke.edu/~ccc14/sta-663/LinearAlgebraMatrixDecompWithSolutions.html
Matrix decompositions are an important step in solving linear systems in a computationally efficient manner.
"""

"""
LU Decomposition and Gaussian Elimination

LU decomposition is essentially gaussian elimination, but we work only with the matrix AA (as opposed to the augmented matrix).

Gaussian elimination
Usually, it is more efficient to stop at reduced row eschelon form (upper triangular, with ones on the diagonal), and then use back substitution to obtain the final answer.
partial pivoting = permuting rows
full pivoting = permuting columns

Multiple problems, where the left-hand-side of our matrix equation does not change, but there are many outcome vectors bb. In this case, it is more efficient to decompose A.
-> see text
"""
# LU decomposition in Numpy
import numpy as np
import scipy.linalg as la
#np.set_printoptions(suppress=True)
A = np.array([[1,3,4],[2,1,3],[4,1,2]])
print(A)
P, L, U = la.lu(A) # LU decomposition
print(np.dot(P.T, A)) # P is pivoting index, so that P^T * A = A.after.pivoting
# -> see a partial pivoting here (to increase numerical stability)
print(np.dot(L, U)) # checking that P = L*U
print(P)
print(L)
print(U)

"""
Cholesky Decomposition
A is summetric, positive-definite matrix:
  A = A^T
  there's a unique decompostion A = L*L^T
  L is kind of square root of A
"""
# Cholesky Decomposition
A = np.array([[1,3,5],[3,13,23],[5,23,42]])
L = la.cholesky(A) # twice as fast as LU
print(np.dot(L.T, L)) # checking, it's back to A
print(L)
print(A)

"""
Matrix Decompositions for PCA and Least Squares
http://people.duke.edu/~ccc14/sta-663/LinearAlgebraMatrixDecompWithSolutions.html#matrix-decompositions-for-pca-and-least-squares

Eigendecomposition
Eigenvector of a matrix A is a non-zero vector v such that
Av=λv
for some scalar λ
The value λ is called an eigenvalue of A.

If an n×n matrix A has n linearly independent eigenvectors, then A may be decomposed in the following manner:
A = B*Λ*B^-1
where Λ is a diagonal matrix whose diagonal entries are the eigenvalues of A and the columns of B are the corresponding eigenvectors of A.

An n×n matrix is diagonizable ⟺ it has n linearly independent eigenvectors.

A symmetric, positive definite matrix has only positive eigenvalues and its eigendecomposition
A = B*Λ*B^-1
is via an orthogonal transformation B. (i.e. its eigenvectors are an orthonormal set).

Calculating eigenvalues
From definition
Av − λI = 0
where I is the identity matrix of dimension n and 0 is an n-dimensional zero vector. Therefore, the eigenvalues of A satisfy:
det(A − λI) = 0
The left-hand side above is a polynomial in λ, and is called the characteristic polynomial of A. Thus, to find the eigenvalues of A, we find the roots of the characteristic polynomial.
In practice, numerical methods are used - both to find eigenvalues and their corresponding eigenvectors.
"""

# Eigendecomposition
A = np.array([[0,1,1],[2,1,0],[3,4,5]])
u, V = la.eig(A)
print(np.dot(V,np.dot(np.diag(u), la.inv(V))))
print(u)

# Many matrices are not diagonizable, and many have complex eigenvalues (even if all entries are real).
A = np.array([[0,1],[-1,0]])
print(A)
u, V = la.eig(A)
print(np.dot(V,np.dot(np.diag(u), la.inv(V))))
print(u)

# If you know the eigenvalues must be real because A is a positive definite (e.g. covariance) matrix use real_if_close
A = np.array([[0,1,1],[2,1,0],[3,4,5]])
u, V = la.eig(A)
print(u)
print(np.real_if_close(u))

"""
Singular values
For any m×n matrix A, we define its singular values to be the square root of the eigenvalues of A^T*A. These are well-defined as A^T*A is always symmetric, positive-definite, so its eigenvalues are real and positive.
- mapping sphere into an ellipse
- stability of the matrix

QR decompositon
a method to write a matrix AA as the product of two matrices of simpler form
A = Q*R
where Q is an m×nm×n matrix with Q*Q^T = I (i.e. Q is orthogonal) and R is an n×n upper-triangular matrix.
This is the matrix form of the Gram-Schmidt orthogonalization of the columns of A.
"""


"""
Singular Value Decomposition
http://people.duke.edu/~ccc14/sta-663/LinearAlgebraMatrixDecompWithSolutions.html#singular-value-decomposition

For any m×n matrix A, we may write:
A=U*D*V
  U is a unitary (orthogonal in the real case) m×m matrix
  D is a rectangular, diagonal m×nn matrix with diagonal entries d1,...,dmd1,...,dm all non-negative
  V is a unitary (orthogonal) n×n matrix
SVD is used in principle component analysis and in the computation of the Moore-Penrose pseudo-inverse.
"""

"""
Stabilty and Condition Number
"""
import numpy as np
import scipy.linalg as la
A = np.array([[8,6,4,1],[1,4,5,1],[8,4,1,1],[1,4,3,6]])
b = np.array([19,11,14,14])
la.solve(A,b)
b = np.array([19.01,11.05,14.07,14.05]) # just a little bit different values...
la.solve(A,b) # ... and very different results
"""
We say matrix A is ill-conditioned.
This happens when a matrix is ‘close’ to being singular (i.e. non-invertible).

Condition Number
A measure of this type of behavior is called the condition number. It is defined as:

cond(A) = ||A||*||A^-1||

In general, it is difficult to compute.
Fact:
cond(A) = λ1 / λn
λ1 is the maximum singular value of A
λn is the smallest
The higher the condition number, the more unstable the system. In general if there is a large discrepancy between minimal and maximal singular values, the condition number is large.
"""
U, s, V = np.linalg.svd(A)
print(s)
print(max(s)/min(s)) # the condition number is high for A
"""
Preconditioning

Instead of solving
A*x = b
we solve
D^-1 * A*x = D^-1 * b
and the condition number for D^-1 * A is lower than for A.
"""




"""
Change of Basis
http://people.duke.edu/~ccc14/sta-663/PCASolutions.html
"""

# Variance and covariance
def cov(x, y):
    """Returns covariance of vectors x and y)."""
    xbar = x.mean()
    ybar = y.mean()
    return np.sum((x - xbar)*(y - ybar))/(len(x) - 1)
X = np.random.random(10)
Y = np.random.random(10)
np.array([[cov(X, X), cov(X, Y)], [cov(Y, X), cov(Y,Y)]])
# This can of course be calculated using numpy's built in cov() function
np.cov(X, Y)
# Extension to more vairables is done in a pair-wise way
Z = np.random.random(10)
np.cov([X, Y, Z])

# Eigendecomposition of the covariance matrix
mu = [0,0]
sigma = [[0.6,0.2],[0.2,0.2]]
n = 1000
x = np.random.multivariate_normal(mu, sigma, n).T
plt.scatter(*x, alpha=0.3)
A = np.cov(x)
A
e, v = np.linalg.eig(A)
e
v
list(zip(e, v.T))
# plt.scatter(x[0,:], x[1,:], alpha=0.2)
plt.scatter(*x, alpha=.2) # better
for e_, v_ in zip(e, v.T):
    plt.plot([0, 3*e_*v_[0]], [0, 3*e_*v_[1]], 'r-', lw=2)
    # -> 3 * eigenvalue * eigenvector, plotted from (0,0) point
# plt.axis([-3,3,-3,3]) # resizing of the plotting area, not necessary
plt.title('Eigenvectors of covariance matrix scaled by eigenvalue.')

# this is something
m = np.array([[1,2,3],[6,5,4]])
m
ms = m - m.mean(1).reshape(2,1)
ms
np.dot(ms, ms.T)/2

"""
PCA
http://people.duke.edu/~ccc14/sta-663/PCASolutions.html#pca

Principal Components Analysis (PCA) basically means to find and rank all the eigenvalues and eigenvectors of a covariance matrix. This is useful because high-dimensional data (with p features) may have nearly all their variation in a small number of dimensions k, i.e. in the subspace spanned by the eigenvectors of the covariance matrix that have the k largest eigenvalues. If we project the original data into this subspace, we can have a dimension reduction (from p to k) with hopefully little loss of information.

Numerically, PCA is typically done using SVD (Singular Value Decomposition) on the data matrix rather than eigendecomposition on the covariance matrix. The next section explains why this works.
"""

# for zero mean feature vectors Cov = X * X^T / (n-1)
x
x.mean(1) # it's not 0, but close
x.mean(1)[:,None] # prepared to centralize the features
x.mean(1).reshape(2,1) # ditto
(x - x.mean(1)[:,None]).mean(1) # 0 mean
x = x - x.mean(1)[:,None]
# eigendecomposition
e1, v1 = np.linalg.eig(np.dot(x, x.T)/(n-1))
e1
v1
list(zip(e1,v1.T))
# plotting
plt.scatter(*x, alpha=0.2)
for e_, v_ in zip(e1, v1.T):
    plt.plot([0, 3*e_*v_[0]], [0, 3*e_*v_[1]], 'r-', lw=2)
plt.axis([-3,3,-3,3])

"""
Change of basis via PCA

We can transform the original data set so that the eigenvectors are the basis vectors and find the new coordinates of the data points with respect to this new basis
"""

# the covariance matrix is a real symmetric matrix, and so the eigenvector matrix is an orthogonal matrix
np.cov(x)
e, v = np.linalg.eig(np.cov(x))
v.dot(v.T) # = I
# -> so this is orthogonal matrix, i.e. the eigenvectors are orthogonal
list(zip(e, v.T))
v
v.T
np.linalg.inv(v)


"""
Linear algebra review for change of basis
basis vectors: u, v for B, u', v' for B'
w'=(x',y') in B' = w in B
w = P * w', w' = P^-1 * w

For eigenvectors V, V^T = V^-1 (V is orthogonal, it's a matrix eigenvectors as columns)
w' = V^T * w
"""
mu = [0,0]
sigma = [[0.6,0.2],[0.2,0.2]]
n = 1000
x = np.random.multivariate_normal(mu, sigma, n).T
e1, v1 = np.linalg.eig(np.cov(x))
ys = np.dot(v1.T, x)
# -> converting vectors x (in standard basis B) into ys vectors in the basis given by the eigenvectors v1 (space B')
np.eye(2) == np.dot(v1.T, v1)
np.isclose(np.eye(2), np.dot(v1.T, v1))
# -> eigenvectors translated into basis given by the eigenvectors = identity matrix
#    also it's an orthogonal matrix so v1^T * v1 = I
v1.T == np.linalg.inv(v1) # it's an orthogonal matrix
np.isclose(v1.T, np.linalg.inv(v1)) # OK
# plot in the eigenvector basis
plt.scatter(*ys, alpha=0.2); plt.axis([-3,3,-3,3]) # to preserve aspect ratio
for e_, v_ in zip(e1, np.eye(2)):
    # np.eye(2) == np.dot(v1.T, v1)
    plt.plot([0, 3*e_*v_[0]], [0, 3*e_*v_[1]], 'r-', lw=2)
plt.axis([-3,3,-3,3])
"""
If we only use the first column of ys, we will have the projection of the data onto the first principal component, capturing the majoriyt of the variance in the data with a single featrue that is a linear combination of the original features.

We may need to transform the (reduced) data set to the original feature coordinates for interpretation. This is simply another linear transform (matrix multiplication).
"""
# plot back in the original basis
zs = np.dot(v1, ys)
plt.scatter(*zs, alpha=.2)
for e_, v_ in zip(e1, v1.T):
    plt.plot([0, 3*e_*v_[0]], [0, 3*e_*v_[1]], 'r-', lw=2)
plt.axis([-3,3,-3,3])

# some experiments to replace the loop, but I gave up
for e_, v_ in zip(e1, v1.T):
    print([0, 3*e_*v_[0]], [0, 3*e_*v_[1]])
np.vstack(((3*e1*v1).T.flatten(),np.zeros(4)))
#####################################################

u, s, v = np.linalg.svd(x)
u.dot(u.T) # identity matrix

"""
Dimension reduction via PCA (Principal Component Analysis)

We have the sepectral decomposition of the covariance matrix A
A = Q^-1 * Λ * Q

Suppose Λ is a rank p matrix. To reduce the dimensionality to k ≤ p, we simply set all but the first k values of the diagonal of Λ to zero. This is equivvalent to ignoring all except the first k principal componnents.

What does this achieve? Recall that A is a covariance matrix, and the trace of the matrix is the overall variability, since it is the sum of the variances.
"""
A = np.cov(x)
A
A.trace()
e, v = np.linalg.eig(A)
list(zip(e,v))
D = np.diag(e)
D # a diagonal matrix with eigenvalues at the diagonal
D.trace() # this equals total variability (sum of variances)
A.trace() == D.trace() # total variability
np.isclose(A.trace(), D.trace())
D[0,0] / D.trace() # variability explained by the first eigenvalue
D[1,1] / D.trace() # and by the second
"""
Since the trace is invariant under change of basis, the total variability is also unchaged by PCA. By keeping only the first k principal components, we can still “explain”
∑,i=1..k,e[i] / ∑e
of the total variability. Sometimes, the degree of dimension reduction is specified as keeping enough principal components so that (say) 90% fo the total variability is exlained.
"""

"""
Using Singular Value Decomposition (SVD) for PCA

SVD is a decomposition of the data matrix
X = U * S * V^T
where U and V are orthogonal matrices and S is a diagnonal matrix.

Recall that the transpose of an orthogonal matrix is also its inverse, so if we multiply on the right by X^T, we get the follwoing simplification:

Compare with the eigendecomposition of a matrix
A = W * Λ * W^-1
(compare it with PCA -> A is the cov(X))
we see that SVD gives us the eigendecomposition of the matrix X*X^T, which as we have just seen, is basically a scaled version of the covariance for a data matrix with zero mean, with the eigenvectors given by U and eigenvalues by S^2 (scaled by n−1).
"""
u, s, v = np.linalg.svd(x) # SVD decomposition
e2 = s**2/(n-1) # eigenvalues
v2 = u # eigenvectors
# plot in the standard basis
plt.scatter(*x, alpha=.2)
for e_, v_ in zip(e2, v2):
    plt.plot([0, 3*e_*v_[0]], [0, 3*e_*v_[1]], 'r-', lw=2)
plt.axis([-3,3,-3,3])
# plot in the eigenvector basis
plt.scatter(*np.dot(v2.T,x), alpha=.2) # x transformed into eigenvector base
for e_, v_ in zip(e2, np.dot(v2.T,v2)): # eigenvectors transformed into eigvec. base (so identity matrix)
    plt.plot([0, 3*e_*v_[0]], [0, 3*e_*v_[1]], 'r-', lw=2)
plt.axis([-3,3,-3,3])
# comparison with using cov() and linalg.eig(), the results are very close
v1 # from eigenvectors of covariance matrix
v2 # from SVD - very close in abs(), in v2 the first column is (first vector) is multiplied by (-1)
e1 # from eigenvalues of covariance matrix
e2 # from SVD - very close

