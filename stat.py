# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:36:58 2017
Computational Statistics in Python
http://people.duke.edu/~ccc14/sta-663/index.html
(for exercises - see a separate file)
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





"""
Optimization and Non-linear Methods
http://people.duke.edu/~ccc14/sta-663/OptimizationInOneDimension.html

Example: Maximum Likelihood Estimation (MLE)
"""

"""
Bisection Method (f is continuous)

The bisection method is one of the simplest methods for finding zeroes of a non-linear function. It is guaranteed to find a root - but it can be slow. The main idea comes from the intermediate value theorem
"""
def f(x):
    return x**3 + 4*x**2 -3
# division 1
x = np.linspace(-3.1, 0, 100)
plt.plot(x, x**3 + 4*x**2 -3)
a = -3.0
b = -0.5
c = 0.5*(a+b)
plt.text(a,-1,"a")
plt.text(b,-1,"b")
plt.text(c,-1,"c")
plt.scatter([a,b,c], [f(a), f(b),f(c)], s=50)#, facecolors='none')
plt.scatter([a,b,c], [0,0,0], s=50, c='red')
xaxis = plt.axhline(0)
# division 2
x = np.linspace(-3.1, 0, 100)
plt.plot(x, x**3 + 4*x**2 -3)
d = 0.5*(b+c)
plt.text(d,-1,"d")
plt.text(b,-1,"b")
plt.text(c,-1,"c")
plt.scatter([d,b,c], [f(d), f(b),f(c)], s=50)#, facecolors='none')
plt.scatter([d,b,c], [0,0,0], s=50, c='red')
xaxis = plt.axhline(0)
"""
Secant Method (f is continuous)

also begins with two initial points, but without the constraint that the function values are of opposite signs. We use the secant line to extrapolate the next candidate point. The secant method has a convergence rate that is faster than linear, but not quite quadratic (x^1.6).
"""
def f(x):
    return (x**3-2*x+7)/(x**4+2)
x = np.arange(-3, 5, 0.1);
y = f(x)
plt.plot(x, y)
plt.xlim(-3, 4)
plt.ylim(-.5, 4)
plt.xlabel('x')
plt.axhline(0)
t = np.arange(-10, 5., 0.1)
x0=-1.2
x1=-0.5
xvals = []
xvals.append(x0)
xvals.append(x1)
notconverge = 1
count = 0
cols=['r--','b--','g--','y--']
while (notconverge==1 and count <  3):
    slope=(f(xvals[count+1])-f(xvals[count]))/(xvals[count+1]-xvals[count])
    # -> slope = [f(x1)-f(x0)] / [x1-x0]
    #    change of y by x+=1
    intercept=-slope*xvals[count+1]+f(xvals[count+1])
    # -> y-intercept = f(x1) - slope*x1
    #    imagine on a plot: vertical distance between intercept level & f(x1) level is slope*x1, as the line of interest passes (0,intercept) and (x1, f(x1)). To find the absolut level of the intercept (i.e. distance between x axis and intercept) you can deduct slope*x1 from f(x1).
    plt.plot(t, slope*t + intercept, cols[count])
    # -> plot the secant line
    nextval = -intercept/slope
    # -> x-intercept = - y-intercept / slope
    #    imagine a triangle (x-intercept, 0) - (0, 0)- (0, y-intercept)
    #    then the slope is - y-intercept/x-intercept
    #    rearrange that and you get the x-intercept
    if abs(f(nextval)) < 0.001:
        notconverge=0
    else:
        xvals.append(nextval)
    count = count+1
plt.show()
# convergence is done in order: red > blue > green (> yellow)
"""
Newton-Rhapson Method (f is continuous & differentiable)

We want to find the value θ so that some (differentiable) function g(θ)=0. Idea: start with a guess, θ0. Let θ~ denote the value of θ for which g(θ)=0 and define h=θ~−θ0.
This implies that
h ≈ g(θ0) / g′(θ0)
-> imagine a triangle having following sides: h, g(θ0), hypotenuse
    h is horizontal, g(θ0) is vertical
    hypotenuse has slope = g'(θ0) = g(θ0) / h
    so then h = g(θ0) / g′(θ0)

So that
θ~ ≈ θ0  −  g(θ0) / g′(θ0)
-> If slope is + and g(θ0) (the height of the triangle) is +, then the other verticle (the θ1 = potential θ~, where the hypothenuse intersects the x-axis) lays to the left of θ0. So to get you need to deduct h from θ0:
    so θ~ ≈ θ1 = θ0 - h = θ0  −  g(θ0) / g′(θ0)

Thus, we set our next approximation
θ1 = θ0  −  g(θ0) / g′(θ0)
and we have developed an interative procedure with
θn = θn−1  −  g(θn−1) / g′(θn−1)
Convergence is quadratic.
-> The only difference from the secant method is that instead of taking two points & calculating the slope, here we take just one point and take the slope to be the derivative at this point. All else is the same, i.e. the next point to consider is at the intersection of hypothenuse and x-axis.
"""
# example function
x = np.arange(-5,5, 0.1);
y = (x**3-2*x+7)/(x**4+2)
plt.plot(x, y)
plt.xlim(-4, 4)
plt.ylim(-.5, 4)
plt.xlabel('x')
plt.axhline(0)
plt.title('Example Function')
plt.show()
# convergence by Newton-Rhapson
x = np.arange(-5,5, 0.1);
y = (x**3-2*x+7)/(x**4+2)
plt.plot(x, y)
plt.xlim(-4, 4)
plt.ylim(-.5, 4)
plt.xlabel('x')
plt.axhline(0)
plt.title('Good Guess')
t = np.arange(-5, 5., 0.1)
x0=-1.5
# -> here we make a very good guess of -1.5 (say from graph inspection)
xvals = []
xvals.append(x0)
notconverge = 1
count = 0
cols=['r--','b--','g--','y--','c--','m--','k--','w--']
# You could rewrite the loop defining the function and its derivative as python functions
#def func(x):
#    return (x**3-2*x+7)/(x**4+2)
while (notconverge==1 and count <  6):
    funval=(xvals[count]**3-2*xvals[count]+7)/(xvals[count]**4+2)
    # -> value of the function at xi
    slope=-((4*xvals[count]**3 *(7 - 2 *xvals[count] + xvals[count]**3))/(2 + xvals[count]**4)**2) + (-2 + 3 *xvals[count]**2)/(2 + xvals[count]**4)
    # -> value of the derivative at xi = slope
    intercept=-slope*xvals[count]+(xvals[count]**3-2*xvals[count]+7)/(xvals[count]**4+2)
    # -> the y-intercept = value of the function at xi - slope*xi (imagine this on a graph)
    plt.plot(t, slope*t + intercept, cols[count])
    # -> the tangent line indicating new candidate for θ~ at its intersection with x-axis
    nextval = -intercept/slope
    # the new candidate for θ~ = the intercept of the tangent line and x-axis
    if abs(funval) < 0.01:
        notconverge=0
    else:
        xvals.append(nextval)
    count = count+1
plt.show()
funval # this is how close to 0 we got
xvals # -2.24
xvals[count-1]
# convergence is done in order: red > blue > green > yellow

"""
Fatal flaw of both Newton-Rhapson and Secant methods - thay fail when encounter a horizontal asymptote.
"""
x = np.arange(-5, 5, 0.1);
y = (x**3-2*x+7)/(x**4+2)
plt.plot(x, y)
plt.xlim(-4, 4)
plt.ylim(-.5, 4)
plt.xlabel('x')
plt.axhline(0)
plt.title('Bad Guess')
t = np.arange(-5, 5., 0.1)
x0=-0.5
xvals = []
xvals.append(x0)
notconverge = 1
count = 0
cols=['r--','b--','g--','y--','c--','m--','k--','w--']
while (notconverge==1 and count <  6):
    funval=(xvals[count]**3-2*xvals[count]+7)/(xvals[count]**4+2)
    slope=-((4*xvals[count]**3 *(7 - 2 *xvals[count] + xvals[count]**3))/(2 + xvals[count]**4)**2) + (-2 + 3 *xvals[count]**2)/(2 + xvals[count]**4)
    intercept=-slope*xvals[count]+(xvals[count]**3-2*xvals[count]+7)/(xvals[count]**4+2)
    plt.plot(t, slope*t + intercept, cols[count])
    nextval = -intercept/slope
    if abs(funval) < 0.01:
        notconverge = 0
    else:
        xvals.append(nextval)
    count = count+1
plt.show()
funval # small
notconverge # 0
xvals # -102
# -> it started to look for the minimum point towards the expansion of the function around a horizontal asymptote (it is lim.x->-inf.f(x) = 0), and stopped at a point where f(x) was sufficiently close to 0 (x = -102). This is wrong as there's now minimum out there.
"""
Basins of Attraction Can Be ‘Close’
"""
# convergence 1
def f(x): # function
    return x**3 - 2*x**2 - 11*x +12
def s(x): # its derivative
    return 3*x**2 - 4*x - 11
x = np.arange(-5,5, 0.1);
plt.plot(x, f(x))
plt.xlim(-4, 5)
plt.ylim(-20, 22)
plt.xlabel('x')
plt.axhline(0)
plt.title('Basin of Attraction')
t = np.arange(-5, 5., 0.1)
x0=2.43
xvals = []
xvals.append(x0)
notconverge = 1
count = 0
cols=['r--','b--','g--','y--','c--','m--','k--','w--']
while (notconverge==1 and count <  8):
    funval = f(xvals[count])
    slope = s(xvals[count])
    intercept=-slope*xvals[count]+funval
    plt.plot(t, slope*t + intercept, cols[count])
    nextval = -intercept/slope
    if abs(funval) < 0.01:
        notconverge = 0
    else:
        xvals.append(nextval)
    count = count+1
plt.show()
xvals
xvals[count-1]
funval
# -> it found -3.17 which is close to a root that a bit further away from the starting point. There are two roots that are closer; if max count increased from 6 to 8, then it coneverges to -3

# convergence 2
plt.plot(x, f(x))
plt.xlim(-4, 5)
plt.ylim(-20, 22)
plt.xlabel('x')
plt.axhline(0)
plt.title('Basin of Attraction')
t = np.arange(-5, 5., 0.1)
x0=2.349
xvals = []
xvals.append(x0)
notconverge = 1
count = 0
cols=['r--','b--','g--','y--','c--','m--','k--','w--']
while (notconverge==1 and count <  6):
    funval = f(xvals[count])
    slope = s(xvals[count])
    intercept=-slope*xvals[count]+funval
    plt.plot(t, slope*t + intercept, cols[count])
    nextval = -intercept/slope
    if abs(funval) < 0.01:
        notconverge = 0
    else:
        xvals.append(nextval)
    count = count+1
plt.show()
xvals
xvals[count-1]
# -> now it founded a closer root at 1.0
"""
Convergence rate for Newton-Rhapson (NR) method - [...]
"""
"""
Inverse Quadratic Interpolation

a polynomial interpolation = find the polynomial of least degree that fits a set of points
Inverse quadratic interpolation = means we do quadratic interpolation on the inverse function
Convergence rate is approximately 1.8
"""
"""
Brent's Method

a combination of bisection, secant and inverse quadratic interpolation
a ‘bracketed’ method i.e. starts with points (a,b) such that f(a)f(b)<0 (like bisection)
attempts to assess when interpolation will go awry, and if so, performs a bisection step
it has certain criteria to reject an iterate
the default method that scypy uses to minimize a univariate function
"""
from scipy.optimize import minimize_scalar
def f(x):
    return (x - 2) * x * (x + 2)**2
# finding the minimum point
res = minimize_scalar(f)
res.x
res
# plot
x = np.arange(-5,5, 0.1);
plt.plot(x, f(x))
plt.xlim(-4, 4)
plt.ylim(-12, 20)
plt.xlabel('x')
plt.axhline(0)
plt.axvline(res.x, c='r', ls='--')
plt.axhline(res.fun, c='r', ls='--')
# finiding zeros/roots
plt.axvline(scipy.optimize.brentq(f, -1, .5), c='g', ls=':')
plt.axvline(scipy.optimize.brentq(f, .5, 3), c='g', ls=':')
plt.axvline(scipy.optimize.newton(f, -3), c='g', ls=':')




"""
Practical Optimizatio Routines
http://people.duke.edu/~ccc14/sta-663/BlackBoxOptimization.html
"""

"""
Finding roots
We generally need to proivde a starting point in the vicinitiy of the root.
For iD root finding, this is often provided as a bracket (a, b) where a and b have opposite signs.
"""
# Univariate roots and fixed points
def f(x):
    return x**3-3*x+1
x = np.linspace(-3,3,100)
plt.axhline(0)
plt.plot(x, f(x), 'r');

# Finding roots
from scipy.optimize import brentq, newton
brentq(f, -3, 0), brentq(f, 0, 1), brentq(f, 1,3)
# plot
plt.axhline(0)
plt.plot(x, f(x), 'r')
for i in [brentq(f, -3, 0), brentq(f, 0, 1), brentq(f, 1, 3)]:
    # -> 3 different brackets (ranges) to look for a root point
    plt.axvline(i, c='g', ls='--')

# Finding fixed points
# Finding the fixed points of a function g(x)=x is the same as finding the roots of g(x)−x. However, specialized algorihtms also exist - e.g. using scipy.optimize.fixedpoint
from scipy.optimize import fixed_point
def f(x, r):
    """Discrete logistic equation."""
    return r*x*(1-x)
n = 100
fps = np.zeros(n)
for i, r in enumerate(np.linspace(0, 4, n)):
    fps[i] = fixed_point(f, 0.5, args=(r, ))
plt.plot(np.linspace(0, 4, n), fps);
# -> fixed points for this are: x = 0 & x = 1 - 1/r = (r-1)/r, so if r < 1 then fixed point is (negative, 0), if r>1 then (0, positive). It looks like the fixed_point function found only the higher of the two fixed points.
plt.scatter(np.linspace(0, 4, n), fps);
# checking
[(i,r) for i, r in enumerate(np.linspace(0, 4, n))]
fps
# what if different starting point
for i, r in enumerate(np.linspace(0, 4, n)):
    fps[i] = fixed_point(f, -0.4, args=(r, ))
plt.plot(np.linspace(0, 4, n), fps);
# -> strange results

# Mutlivariate roots and fixed points
from scipy.optimize import root, fsolve
def f(x):
    return [x[1] - 3*x[0]*(x[0]+1)*(x[0]-1),
            .25*x[0]**2 + x[1]**2 - 1]
sol = root(f, (0.5, 0.5))
sol
f(sol.x) # (0,0)
sol = root(f, (12,12))
sol
f(sol.x) # (0,0)

x = np.linspace(-1,2,100)
X = np.vstack([x, x])
plt.plot(X, f(X))
# -> this plot doesn't make much sense, it shows vertical segments (x, y1-y2), where (y1,y2) = f(x1,x2) and x1 = x2 = x
# 3d plot
X = np.linspace(-2, 2, 20)
Y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(X, Y)
tmp = np.abs(f((X,Y)))
Z = np.add(tmp[0], tmp[1])
# -> this function returns 2 values; to show points where it returns (0,0) I sum up the absolute values returned into one value >= 0. 0 shows roots
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
fig.colorbar(surf, shrink=0.5, aspect=5)



"""
Optimization Primer

Assumptions
minimize some univariate or multivariate function f(x)
multivariate or real-valued smooth functions (non-smooth, noisy or discrete functions are outside the scope of this course and less common in statistical applications)

Questions
Is the function convex?
Are there any constraints that the solution must meet?

Local vs global optima
optimization methods are nearly always designed to find local optima. For convex problems, there is only one minimum and so this is not a problem. However, if there are multiple local minima, often heuristics such as multiple random starts must be adopted to find a “good” enouhg solution.
"""

# Convexity

# a single global minimum
def f(x):
    return (x-4)**2 + x + 1

with plt.xkcd():
    x = np.linspace(0, 10, 100)
    plt.plot(x, f(x))
    ymin, ymax = plt.ylim()
    plt.axvline(2, ymin, f(2)/ymax, c='red')
    plt.axvline(8, ymin, f(8)/ymax, c='red')
    # -> vertical lines indicating secant & function intersections
    plt.scatter([4, 4], [f(4), f(2) + ((4-2)/(8-2.))*(f(8)-f(2))],
                 edgecolor=['blue', 'red'], facecolor='none', s=100, linewidth=2)
    # -> (4,f(4)) and (4, point at the secant segment)
    plt.plot([2,8], [f(2), f(8)])
    # -> secant segment
    plt.xticks([2,4,8], ('a', 'ta + (1-t)b', 'b'), fontsize=20)
    # -> x ticks
    plt.text(0.2, 40, 'f(ta + (1-t)b) < tf(a) + (1-t)f(b)', fontsize=20)
    # -> the property of convexity
    plt.xlim([0,10])
    plt.yticks([])
    # -> no y ticks
    plt.suptitle('Convex function', fontsize=20)

# Checking if a function is convex using the Hessian
# a twice differntiable function is convex if its Hessian is positive semi-definite, and strictly convex if the Hessian is positive definite
# A univariate function is convex if its second derivative is positive everywhere
from sympy import symbols, hessian, Function, N
import numpy.linalg as la
#import scipy.linalg as la
x, y, z = symbols('x y z')
#f = symbols('f', cls=Function)
f = Function('f')
f = x**2 + 2*y**2 + 3*z**2 + 2*x*y + 2*x*z
f
hessian(f, (x, y, z))
H = np.array(hessian(f, (x, y, z)))
H # see dtype=object
e, v = la.eig(H) # error, see dtype
H.dtype # this is the problem
type(H)
H.values # error, this is for Pandas and returns a numpy array
H2 = H.astype(np.float64)
H2.dtype # ok now
e, v = la.eig(H2)
np.real_if_close(e)
# Since all eigenvalues are positive, the Hessian is positive defintie and the function is convex.

"""
Combining convex functions (useful to determine if more complex functions are covex)
- The intersection of convex functions is convex
- If the functions f and g are convex and a≥0 and b≥0 then the function af+bg is convex
- If the function U is convex and the function g is nondecreasing and convex then the function f defined by f(x)=g(U(x)) is convex

Dealing with Constraints
- convert a problem with constraints into one without constraints
- or use an algorithm that can optimize with constraints
"""

# scipy.optimize
from scipy import optimize as opt
def f(x):
    return x**4 + 3*(x-2)**3 - 15*(x)**2 + 1
x = np.linspace(-8, 5, 100)
plt.plot(x, f(x), c='r')
a = opt.minimize_scalar(f, method='Brent') # default method
b = opt.minimize_scalar(f, method='bounded', bounds=[0, 6])
c = opt.minimize_scalar(f, bounds=[0, 6]) # default Brent ignores bounds
d = opt.minimize_scalar(f, bracket=[2, 3, 4]) # bracket is a starting point, not a bound for the solution
a,b,c,d
plt.plot(x, f(x), c='r')
plt.axvline(a.x, ls='--', c='g', label='a (Brent)')
plt.axvline(b.x, ls=':', c='b', label='b (bounded in [0,6]')
plt.legend()



# Local and global minima
def f(x, offset):
    return -np.sinc(x-offset) # sin(pi*x)/(pi*x)
x = np.linspace(-20, 20, 100)
plt.plot(x, f(x, 5));

# Find minimum point with default setting of the optimizer function
sol = opt.minimize_scalar(f, args=(5,))
# -> note how additional function arguments are passed in
sol
plt.plot(x, f(x,5), 'r', label='function')
plt.axvline(sol.x, ls='--', c='g', label='solution 0')
plt.legend(loc='lower left')
# -> a local but not global minimum found

# Try multiple ranodm starts to find the global minimum
lower = np.random.uniform(-20, 20, 100)
upper = lower + 1
sols = [opt.minimize_scalar(f, args=(5,), bracket=(l, u)) for (l, u) in zip(lower, upper)]
# -> get 100 solutions from different starting points for brent
idx = np.argmin([sol.fun for sol in sols])
# -> select the lowest of the minima
sol = sols[idx]
plt.plot(x, f(x, 5)); plt.axvline(sol.x, c='r', ls='--');

# Using a stochastic algorithm
from scipy.optimize import basinhopping
x0 = 0
sol = basinhopping(f, x0, stepsize=1, minimizer_kwargs={'args': (5,)})
sol
plt.plot(x, f(x, 5)); plt.axvline(sol.x, c='r', ls='--');




"""
Minimizing a multivariate function

Optimization of multivariate scalar functions, where the scalar may (say) be the norm of a vector.

Minimizing a multivariable set of equations f:R^n→R^n is not well-defined, but we will later see how to solve the closely related problem of finding roots or fixed points of such a set of equations.

Rosenbrock “banana” function to illustrate unconstrained multivariate optimization. In 2D:
f(x,y)=b(y−x^2)^2+(a−x)^2
The function has a global minimum at (1,1) and the standard expression takes a=1 and b=100.
"""
"""
Conditinoning of optimization problem

With these values for a and b, the problem is ill-conditioned. As we shall see, one of the factors affecting the ease of optimization is the condition number of the curvature (Hessian). When the codition number is high, the gradient may not point in the direction of the minimum, and simple gradient descent methods may be inefficient since they may be forced to take many sharp turns.
"""
# Rosenbrock (banana) function example - using sympy & Hessian & condition number
from sympy import symbols, hessian, Function, N
x, y = symbols('x y')
f = symbols('f', cls=Function)
a = 1; b = 100
f = b*(y - x**2)**2 + (a - x)**2
# convert into expression that can be numerically evaluted
from sympy import lambdify
func = lambdify([x,y], f, 'numpy')
# plot the functions
import matplotlib.pyplot as plt
X = np.linspace(-3, 3, 100)
Y = np.linspace(-5, 10, 100)
X, Y = np.meshgrid(X, Y)
# plotting the function
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, func(X,Y), cmap=plt.cm.coolwarm)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
# evaluating the Hessian at (1,1)
H = hessian(f, [x, y]).subs([(x,1), (y,1)])
np.array(H)
H.condition_number()
N(H.condition_number()) # high condition number of the Hessian (2508)

# Rosenbrock (banana) function example - contour plotting
def rosen(x):
    """Generalized n-dimensional version of the Rosenbrock function"""
    return sum(100*(x[1:]-x[:-1]**2.0)**2.0 +(1-x[:-1])**2.0)
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = rosen(np.vstack([X.ravel(), Y.ravel()])).reshape((100,100))
# Note: the global minimum is at (1,1) in a tiny contour island
plt.contour(X, Y, Z, np.arange(10)**5)
plt.text(1, 1, 'x', va='center', ha='center', color='red', fontsize=20);

"""
Gradient Descent

The gradient (or Jacobian) at a point indicates the direction of steepest ascent. Since we are looking for a minimum, one obvious possibility is to take a step in the opposite direction to the graident. We weight the size of the step by a factor α known in the machine learning literature as the learning rate. If α is small, the algorithm will eventually converge towards a local minimum, but it may take long time. If α is large, the algorithm may converge faster, but it may also overshoot and never find the minimum.

Some algorithms also determine the appropriate value of α at each stage by using a line search, i.e.,
α∗ = argmin (α) f(x.k − α*∇f(x.k))
(find alpha that makes a step to the minimum point of function f in the given direction)

The problem is that the gradient may not point towards the global minimum especially when the condition number is large, and we are forced to use a small α for convergence. Becasue gradient descent is unreliable in practice, it is not part of the scipy optimize suite of functions, but we will write a custom function below to ilustrate how it works.
"""
import numpy.linalg as la
from scipy import optimize as opt
# Rosenbrock (banana) function
# see just above & run
def rosen_der(x):
    """Derivative of generalized Rosen function."""
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der
def custmin(fun, x0, args=(), maxfev=None, alpha=0.0002,
        maxiter=100000, tol=1e-10, callback=None, **options):
    """Implements simple gradient descent for the Rosen function."""
    bestx = x0
    besty = fun(x0)
    funcalls = 1
    niter = 0
    improved = True
    stop = False
    while improved and not stop and niter < maxiter:
        niter += 1
        # the next 2 lines are gradient descent
        step = alpha * rosen_der(bestx)
        bestx = bestx - step
        besty = fun(bestx)
        funcalls += 1
        if la.norm(step) < tol:
            improved = False
        if callback is not None:
            callback(bestx)
        if maxfev is not None and funcalls >= maxfev:
            stop = True
            break
    return opt.OptimizeResult(fun=besty, x=bestx, nit=niter,
                              nfev=funcalls, success=(niter > 1))
def reporter(p):
    """Reporter function to capture intermediate states of optimization."""
    global ps
    ps.append(p)
# Initial starting position
x0 = np.array([4,-4.1])
ps = [x0]
opt.minimize(rosen, x0, method=custmin, callback=reporter)
opt.minimize(rosen, (1,1), method=custmin, callback=reporter) # reports success: False if starting at minimum
# -> minimize using gradient descent (as defined in custmin) + record the steps by reporter
ps = np.array(ps)
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.contour(X, Y, Z, np.arange(10)**5)
plt.plot(ps[:, 0], ps[:, 1], '-o')
plt.subplot(122)
plt.semilogy(range(len(ps)), rosen(ps.T));


"""
Newton’s method and variants

To find a minimum -> find root of the derivative -> find root using Newton's method generalized to n dimensions (Jacobian becomes Hessian).
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy import optimize as opt
from scipy.optimize import rosen, rosen_der, rosen_hess

# Rosenbrock (banana) function example - contour plotting
def rosen(x):
    """Generalized n-dimensional version of the Rosenbrock function"""
    return sum(100*(x[1:]-x[:-1]**2.0)**2.0 +(1-x[:-1])**2.0)
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = rosen(np.vstack([X.ravel(), Y.ravel()])).reshape((100,100))

# Initial starting position
x0 = np.array([4,-4.1])
ps = [x0]
opt.minimize(rosen, x0, method='Newton-CG', jac=rosen_der, hess=rosen_hess, callback=reporter)
ps = np.array(ps)
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.contour(X, Y, Z, np.arange(10)**5)
plt.plot(ps[:, 0], ps[:, 1], '-o')
plt.subplot(122)
plt.semilogy(range(len(ps)), rosen(ps.T));
"""
As calculating the Hessian is computationally expensive, first order methods only use the first derivatives. Quasi-Newton methods use functions of the first derivatives to approximate the inverse Hessian. A well know example of the Quasi-Newoton class of algorithjms is BFGS, named after the initials of the creators. As usual, the first derivatives can either be provided via the jac= argument or approximated by finite difference methods.
"""
ps = [x0]
opt.minimize(rosen, x0, method='BFGS', callback=reporter)
ps = np.array(ps)
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.contour(X, Y, Z, np.arange(10)**5) # you should plot the data into bigger range
plt.plot(ps[:, 0], ps[:, 1], 'r-o')
plt.subplot(122)
plt.semilogy(range(len(ps)), rosen(ps.T));

# below more X/Y points added
#x = np.linspace(-5, 7, 100)
#y = np.linspace(-5, 40, 100)
#X, Y = np.meshgrid(x, y)
#Z = rosen(np.vstack([X.ravel(), Y.ravel()])).reshape((100,100))
#plt.figure(figsize=(12,4))
#plt.subplot(121)
#plt.contour(X, Y, Z, np.arange(10)**5) # you should plot the data into bigger range
#plt.plot(ps[:, 0], ps[:, 1], 'r-o')
#plt.subplot(122)
#plt.semilogy(range(len(ps)), rosen(ps.T));
"""
Finally, there are some optimization algorithms not based on the Newton method, but on other heuristic search strategies that do not require any derivatives, only function evaluations.

One well-known example is the Nelder-Mead simplex algorithm.
"""
ps = [x0]
opt.minimize(rosen, x0, method='nelder-mead', callback=reporter)
ps = np.array(ps)
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.contour(X, Y, Z, np.arange(10)**5)
plt.plot(ps[:, 0], ps[:, 1], '-o')
plt.subplot(122)
plt.semilogy(range(len(ps)), rosen(ps.T));



"""

Constrained optimization

http://people.duke.edu/~ccc14/sta-663/BlackBoxOptimization.html#constrained-optimization
https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize

sometimes the contraint can be incorporated into the goal function (e.g. p>0 by replacing p with e^q and minimizing for q) -> then you can use the unconstrained optimization
otherwise constrained optimization, methods used internally
- constraint violation penalties
- barriers
- Lagrange multipliers
"""

# example from SciPy tutorial
def f(x):
    return -(2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2)
x = np.linspace(0, 3, 100)
y = np.linspace(0, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f(np.vstack([X.ravel(), Y.ravel()])).reshape((100,100))
plt.contour(X, Y, Z, np.arange(-1.99,10, 1)); # loss function
plt.plot(x, x**3, 'k:', linewidth=1) # equality constraint
plt.plot(x, (x-1)**4+2, 'r-', linewidth=1) # inequality constraint
plt.fill([0.5,0.5,1.5,1.5], [2.5,1.5,1.5,2.5], alpha=0.3) # bounds
plt.axis([0,3,0,3])

# setting constraints
"""
inequlaity cosntraint assumes a Cj * x ≥ 0 form
(as usual) the jac is optional and will be numerically estimted if not provided
"""
cons = ({'type': 'eq',
         'fun' : lambda x: np.array([x[0]**3 - x[1]]),
         'jac' : lambda x: np.array([3.0*(x[0]**2.0), -1.0])},
        {'type': 'ineq',
         'fun' : lambda x: np.array([x[1] - (x[0]-1)**4 - 2])})
bnds = ((0.5, 1.5), (1.5, 2.5))

# starting point
x0 = [0, 2.5]

# unconstrained optimization
ux = opt.minimize(f, x0, constraints=None)
ux

# constrained optimization
cx = opt.minimize(f, x0, bounds=bnds, constraints=cons)
cx

# plotting the minimum points (unconstrained and constrained)
#x = np.linspace(0, 3, 100)
#y = np.linspace(0, 3, 100)
#X, Y = np.meshgrid(x, y)
#Z = f(np.vstack([X.ravel(), Y.ravel()])).reshape((100,100))
plt.contour(X, Y, Z, np.arange(-1.99,10, 1));
plt.plot(x, x**3, 'k:', linewidth=1)
plt.plot(x, (x-1)**4+2, 'k:', linewidth=1)
plt.text(ux['x'][0], ux['x'][1], 'x', va='center', ha='center', size=20, color='blue')
plt.text(cx['x'][0], cx['x'][1], 'x', va='center', ha='center', size=20, color='red')
plt.fill([0.5,0.5,1.5,1.5], [2.5,1.5,1.5,2.5], alpha=0.3)
plt.axis([0,3,0,3]);

"""

Curve fitting

http://people.duke.edu/~ccc14/sta-663/BlackBoxOptimization.html#curve-fitting

Case: use non-linear least squares to fit a function to data, perhaps to estimate paramters for a mechanistic or phenomenological model
The curve_fit function uses the quasi-Newton Levenberg-Marquadt aloorithm to perform such fits. Behind the scnees, curve_fit is just a wrapper around the leastsq function that we have already seen in a more conveneint format.
"""
from scipy.optimize import curve_fit

def logistic4(x, a, b, c, d):
    """The four paramter logistic function is often used to fit dose-response relationships."""
    return ((a-d)/(1.0+((x/c)**b))) + d

nobs = 24
xdata = np.linspace(0.5, 3.5, nobs)
ptrue = [10, 3, 1.5, 12]
ydata = logistic4(xdata, *ptrue) + 0.5*np.random.random(nobs)

popt, pcov = curve_fit(logistic4, xdata, ydata)
# returns: estimated parameters and var-covar matrix

perr = yerr = np.sqrt(np.diag(pcov))
# table of std.def of the esitmated parameters
print('Param\tTrue\tEstim (+/- 1 SD)')
for p, pt, po, pe  in zip('abcd', ptrue, popt, perr):
    print('%s\t%5.2f\t%5.2f (+/-%5.2f)' % (p, pt, po, pe))
    # prints row by row data for each parameter
# checking
list(zip('abcd', ptrue, popt, perr))

x = np.linspace(0, 4, 100)
y = logistic4(x, *popt)
plt.plot(xdata, ydata, 'o')
plt.plot(x, y);

"""

Finding parameters for ODE models

http://people.duke.edu/~ccc14/sta-663/BlackBoxOptimization.html#finding-paraemeters-for-ode-models

Curve x(t) defined implicitly
dx/dt = -kx
Estimate k and x0 using data.
Sic: this can be solved analytically

"""
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# the ODE describing the curve
def f(x, t, k):
    """Simple exponential decay.
    """
    return -k*x

# the curve defined by solving the ODE
# returns the value of x(t) at a given t, and given the parameters k and x0
def x(t, k, x0):
    """
    Solution to the ODE x’(t) = f(t,x,k) with initial condition x(0) = x0
    """
    x = odeint(f, x0, t, args=(k,))
    # odeint( dx/dt(at t), initial_condition=x(0), additional_parameters_for_dx/dt )
    return x.ravel()

# True parameter values
x0_ = 10
k_ = 0.1*np.pi

# Some random data genererated from closed form solution plus Gaussian noise
ts = np.sort(np.random.uniform(0, 10, 200))
xs = x0_*np.exp(-k_*ts) + np.random.normal(0,0.1,200) # x(t) given in a closed form / explicit t
plt.plot(ts, xs)

popt, cov = curve_fit(x, ts, xs)
# curve_fit( function(x, ...), t_sequence, x(t)_sequence )
k_opt, x0_opt = popt
# optimal parameters: k, x0
print("k = %g" % k_opt)
print("x0 = %g" % x0_opt)
t = np.linspace(0, 10, 100)
plt.plot(ts, xs, '.', t, x(t, k_opt, x0_opt), '-');

"""

Optimization of graph node placement

http://people.duke.edu/~ccc14/sta-663/BlackBoxOptimization.html#optimization-of-graph-node-placement

E.g. using optimization to change the layout of nodes of a graph.
Nodes connected by springs, with some nodes on fixed positions.
Optimization finds the configuraiton of lowest potential energy given that some nodes are fixed (set up as boundary constraints on the positions of the nodes).

the ordination algorithm Multi-Dimenisonal Scaling (MDS) works on a very similar idea
MDS is often used in exploratory analysis of high-dimensional data to get some intuitive understanding of its “structure”

Given
P0 is the initial location of nodes
P is the minimal energy location of nodes given constraints
A is a connectivity matrix - there is a spring between i and j if Aij=1
Lij is the resting length of the spring connecting i and j
In addition, there are a number of fixed nodes whose positions are pinned.
"""
from scipy import optimize as opt
from scipy.spatial.distance import pdist, squareform
n = 20
k = 1 # spring stiffness
P0 = np.random.uniform(0, 5, (n,2))
A = np.ones((n, n))
A[np.tril_indices_from(A)] = 0
L = A.copy()
def energy(P):
    P = P.reshape((-1, 2))
    D = squareform(pdist(P))
    return 0.5*(k * A * (D - L)**2).sum()
energy(P0.ravel())
# fix the position of the first few nodes just to show constraints
fixed = 4
bounds = (np.repeat(P0[:fixed,:].ravel(), 2).reshape((-1,2)).tolist() +
          [[None, None]] * (2*(n-fixed)))
bounds[:fixed*2+4]
sol = opt.minimize(energy, P0.ravel(), bounds=bounds)
plt.scatter(P0[:, 0], P0[:, 1], s=25)
P = sol.x.reshape((-1,2))
plt.scatter(P[:, 0], P[:, 1], edgecolors='red', facecolors='none', s=30, linewidth=2);

"""
Optimization of standard statistical models

http://people.duke.edu/~ccc14/sta-663/BlackBoxOptimization.html#optimization-of-standard-statistical-models


"""

# <-------------------------------------------------------------------------------------------




"""
Fitting ODEs with the Levenberg–Marquardt algorithm
http://people.duke.edu/~ccc14/sta-663/CalibratingODEs.html
"""

"""
Algorithms for Optimization and Root Finding for Multivariate Problems
http://people.duke.edu/~ccc14/sta-663/MultivariateOptimizationAlgortihms.html
"""

"""
Expectation Maximizatio (EM) Algorithm
http://people.duke.edu/~ccc14/sta-663/EMAlgorithm.html
"""

"""
Monte Carlo Methods
http://people.duke.edu/~ccc14/sta-663/MonteCarlo.html
"""

"""
Resampling methods
http://people.duke.edu/~ccc14/sta-663/ResamplingAndMonteCarloSimulations.html
"""





















