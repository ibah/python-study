# -*- coding: utf-8 -*-

import scipy
scipy.__version__
scipy.version.*version?

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# Scipy sub-packages need to be imported separately, e.g.
from scipy import linalg, optimize
# except for the most common functions

# help
help
help(np.abs)
np.info(optimize.fmin)  # doesn't print anything for me...
# looking at the namespace of a module or package
dir(optimize)
# looking at the code
np.source(np.abs)  # doesn't print anyting for me...

''' Basic functions '''

# use of the slicing functionality to provide efficient means for array construction
  # np.mgrid , np.ogrid , np.r_ , np.c_ for quickly constructing arrays
[0]*5
np.concatenate(([1,2],[3,4],[5,6]))
np.r_[-1:1:10j] # ten points, the last point inclusive
  # complex number indicates number of steps -> the last point inclusive
np.r_[-1:1:0.5] # step of 0.5, the last point exclusive
np.concatenate(([3], [0]*5, np.arange(-1, 1.002, 2/9.0)))
# stack arrays by rows (identical shape for all other dimensions required)
np.r_[3,[0]*5,-1:1:10j] # concat into one long row
# stack arrays by cols (identical shape for other dimensions)
np.c_[-1:1:10j]# the same but columns -> 1 col, many rows

# to produce N, N-d arrays which provide coordinate arrays for an N-dimensional volume

np.mgrid[0:1] # 1 element in 1 row (and 1 col)
np.mgrid[0:2] # 2 elements in 1 row and 2 cols
              # [fed by cols]
np.mgrid[0:1,0:1] # 2 elements in 2 slices (each of 1 row & 1 col)
np.mgrid[0:2,0:2] # 8 elements in 2 slices, each of 2 rows and 2 cols
              # [page 0 fed by rows, page 1 fed by cols]
np.mgrid[0:5,0:5] # these are 2 slices/pages, each of 5 rows and 5 columns
              # [page 0 fed by rows, page 1 fed by cols]
np.mgrid[0:4:5j,0:4:5j] # the same but 4 is inclusive & floating type elements
np.mgrid[0:5:4j,0:5:4j]

  # the same using np.meshgrid

x,y = np.meshgrid(np.arange(5),np.arange(5))
x
y
  # now trying to concatenate x and y in the same way as mgrid did:
np.c_[y,x]  # this is 5 rows, each of 10 elements (i.e. 10 columns)
np.hstack((y,x))  # the same
np.concatenate((y,x), axis=1) # the same
np.r_[y,x]  # this is 10 rows, each of 5 elements (i.e. 5 columns)
np.vstack((y,x))  # the same
np.concatenate((y,x), axis=0) # the same
  # finally the correct concatenation by adding a new axis:
from numpy import newaxis
yy = y[newaxis,]
xx = x[newaxis,]
xx
yy
np.concatenate((yy,xx), axis=0) # now the same as mgrid


# Shape manipulation
  # squeezing out length- one dimensions from N-dimensional arrays
  # ensuring that an array is at least 1-, 2-, or 3-dimensional
  # stacking (concatenating) arrays by rows, columns, and “pages “(in the third dimension)
  # splitting arrays (roughly the opposite of stacking arrays)


# Polynomials
# 1-d polynomials
# 1. numpy.poly1d class
from numpy import poly1d
p = poly1d([3,4,5])
print p
print p*p
print p.integ(k=6)
print p.deriv()
p([4, 5])
# 2. an array of coefficients with the first element of the array giving the coefficient of the highest power
  # There are explicit functions to add, subtract, multiply, divide, integrate, differentiate, and evaluate polynomials represented as sequences of coefficients


# Vectorizing functions (vectorize)
  # convert an ordinary Python function which accepts scalars and returns scalars
  # into a “vectorized-function” with the same broadcasting rules as other Numpy functions
def addsubtract(a,b):
   if a > b:
       return a - b
   else:
       return a + b
addsubtract(4,5)
addsubtract(5,4)
addsubtract([4,5],[5,4])  # wrong: it added the lists = concatenated them
addsubtract([5,4],[4,5])  # wrong: it tried to do deduction on lists, but that isn't defined
vec_addsubtract = np.vectorize(addsubtract)
vec_addsubtract([0,3,6,9],[1,3,5,7])  # now correct
  # if the function you have written is the result of some optimization or integration routine,
  # then such functions can likely only be vectorized using vectorize.


# Type handling
  # test for arrays:
np.iscomplex, np.isreal
  # test for objects:
np.iscomplexobj, np.isrealobj
  # for complex, vector, scalars
np.real, np.imag
np.real_if_close  # transforms a complex-valued number with tiny imaginary part into a real number
np.finfo(np.float)
np.finfo(np.float).eps # print out the machine epsilon for floats
np.isscalar  # checking for a scalar
  # for ensuring the proper numpy type is applied:
np.cast?
type(np.pi) # float
np.cast['f'](np.pi) # np.float32

  # Other useful functions
np.angle  # for phase analysis
np.unwrap
np.linspace # return equally spaced samples in a linear or log scale
np.logspace
  # select
np.select(condlist,choicelist,default=0)
  # select is a vectorized form of the multiple if-statement.
  # It allows rapid construction of a function which returns an array of results
  # based on a list of conditions.
  # Each element of the return array is taken from the array in a choicelist
  # corresponding to the first condition in condlist that is true.
x = np.r_[-2:5]
x
np.select([x > 3, x >= 0], [100, x+2], -10)

  # module scipy.misc
import scipy.misc as msc
np.factorial
np.comb # combinations
np.lena # image processing
  # approximating derivatives of functions using discrete-differences
np.central_diff_weights
  # returns weighting coefficients for an equally-spaced NN-point approximation
  # to the derivative of order o

''' Statistics (scipy.stat) '''

from scipy import stats
from scipy.stats import norm

# Random Variables
np.rv_continuous # over 80
np.rv_discrete # over 10
np.info(stats)
# help
print(stats.norm.__doc__)
# finding the support of a distribution
print('bounds of distribution lower: %s, upper: %s' % (norm.a, norm.b))
# all methods and properties of the distribution
dir(norm)
# To obtain the real main methods, we list the methods of the frozen distribution
rv = norm()
dir(rv)
#  the list of available distribution through introspection
#import warnings
#warnings.simplefilter('ignore', DeprecationWarning)























dist_continu = [d for d in dir(stats) if
                isinstance(getattr(stats,d), stats.rv_continuous)]
dist_discrete = [d for d in dir(stats) if
                 isinstance(getattr(stats,d), stats.rv_discrete)]
print 'number of continuous distributions:', len(dist_continu)

print 'number of discrete distributions:  ', len(dist_discrete)
# 86, 13
# The main public methods for continuous RVs:
'''rvs: Random Variates
pdf: Probability Density Function
cdf: Cumulative Distribution Function
sf: Survival Function (1-CDF)
ppf: Percent Point Function (Inverse of CDF)
isf: Inverse Survival Function (Inverse of SF)
stats: Return mean, variance, (Fisher’s) skew, or (Fisher’s) kurtosis
moment: non-central moments of the distribution'''
norm.cdf(0)
norm.cdf([-1., 0, 1]) # passing a list
norm.cdf([-1.96,-1.645,-1.28,0])  # 2.5%, 5%, 10%, 50%
import numpy as np
norm.cdf(np.array([-1., 0, 1])) # passing a numpy array
norm.mean(), norm.std(), norm.var()
norm.stats(moments = "mv") # mean, variance (see more examples)
# To find the median of a distribution we can use the percent point function ppf, which is the inverse of the cdf
norm.ppf(0.5)
norm.ppf([0.025,0.05,0.1,0.5,0.9,0.95,0.975])
# sequence of random variables
norm.rvs(size=3)  # 3 randoms
norm.rvs(3) # no! this is just one random for loc=3
# for reproducibility
np.random.seed(1234) # so all is done using numpy generator
# Relying on a global state is not recommended though. A better way is to use the random_state parameter
norm.rvs(size=5, random_state=1234)  # doesn't work

# Shifting and Scaling
# for all continuous: loc, scale
  # norm
norm.stats(loc = 3, scale = 4, moments = "mv") # mean = 3, std.dev = 4
  # expon
from scipy.stats import expon
expon.mean(scale=3.)  # mean = 1/lambda = scale
  # uniform
from scipy.stats import uniform
uniform.cdf([-1, 0, 0.1, 0.5, 1, 2])
  # support: [0-1), so at 0 is 0%, at 0.5 is 50%, at 1 is 100%
uniform.cdf([0, 1, 2, 3, 4, 5], loc = 1, scale = 4)
  # x -> scale*x + loc = 4*x + 1
  # support: [0,1) -> [0,4) -> [1,5)
  # so 0, 1 give 0%, 2 gives 25%, 3 gives 50%, ...
# We recommend that you set loc and scale parameters explicitly,
# by passing the values as keywords rather than as arguments.
# Repetition can be minimized when calling more than one method of a given RV
# by using the technique of Freezing a Distribution, as explained below.

# Shape parameters
# some distributions require additional shape parameters, e.g. gamma
from scipy.stats import gamma
gamma.numargs
gamma.shapes
gamma(1, scale=2.).stats(moments="mv") # a=1 (exponetial distr.), lambda = 1/scale = 0.5
gamma(a=1, scale=2.).stats(moments="mv") # the same

# Freezing a Distribution
# pass the loc and scale keywords, and other shape params, just once
rv = gamma(1, scale=2.)
rv.mean(), rv.std()

# Broadcasting
# The basic methods pdf and so on satisfy the usual numpy broadcasting rules
# e.g.: upper quantiles (isf) for t distribution of 10 and 11 degrees of freedom
stats.t.isf([0.1, 0.05, 0.01], [[10], [11]])
# the same
stats.t.isf([0.1, 0.05, 0.01], 10)
stats.t.isf([0.1, 0.05, 0.01], 11)
# if the same shape for both arguments, then element matchnig is applied
# e.g. the 10% tail for 10 d.o.f., the 5% tail for 11 d.o.f. and the 1% tail for 12 d.o.f.
stats.t.isf([0.1, 0.05, 0.01], [10, 11, 12]) # 
# checking
stats.t.isf([0.1, 0.05, 0.01], 10)
stats.t.isf([0.1, 0.05, 0.01], [10]) # the same
stats.t.isf([0.1, 0.05, 0.01], [10, 11]) # error, can't broadcast
stats.t.isf([0.1, 0.05, 0.01], [10, 11, 12]) # correct broadcasting: element-wise matching
stats.t.isf([0.1, 0.05, 0.01], [[10], [11]]) # correct broadcasting: cartesian product
stats.t.isf([0.1, 0.05, 0.01], [[10], [11], [12]]) # correct broadcasting: cartesian product

# Specific Points for Discrete Distributions
'''pdf is replaced the probability mass function pmf
  no estimation methods, such as fit, are available
  scale is not a valid keyword parameter
  The location parameter: keyword loc can still be used to shift the distribution.
  The cdf of a discrete distribution is a step function
    hence the inverse cdf, i.e., the percent point function, requires a different definition:
    ppf(q) = min{x : cdf(x) >= q, x integer}'''
# e.g. hypergeometric distribution
from scipy.stats import hypergeom
[M, n, N] = [20, 7, 12]
x = np.arange(4)*2
x
# prb at the kinks of the distribution/the cdf
prb = hypergeom.cdf(x, M, n, N)
prb
hypergeom.ppf(prb, M, n, N) # the same points back
# prb away from the kinks of the distribution/the cdf
hypergeom.ppf(prb + 1e-8, M, n, N)  # here you get next higher integers back
hypergeom.ppf(prb - 1e-8, M, n, N)  # here we get the same integers

# Fitting Distributions
'''The main additional methods of the not frozen distribution are related to
  the estimation of distribution parameters:
    fit: maximum likelihood estimation of distribution parameters, including location and scale
    fit_loc_scale: estimation of location and scale when shape parameters are given
    nnlf: negative log likelihood function
    expect: Calculate the expectation of a function against the pdf or pmf'''
# Caution
'''the distributions have been tested over some range of parameters, however in some corner ranges, a few incorrect results may remain.
the maximum likelihood estimation in fit does not work with default starting parameters for all distributions and the user needs to supply good starting parameters.
Also, for some distribution using a maximum likelihood estimator might inherently not be the best choice'''

# Building Specific Distributions

# Making a Continuous Distribution, i.e., Subclassing rv_continuous
from scipy import stats
class deterministic_gen(stats.rv_continuous):
    def _cdf(self, x):
        return np.where(x < 0, 0., 1.)
    def _stats(self):
        return 0., 0., 0., 0.
deterministic = deterministic_gen(name="deterministic")
deterministic.cdf(np.arange(-3, 3, 0.5))
# pdf is now computed automatically - but beware of the performance issues
deterministic.pdf(np.arange(-3, 3, 0.5))
deterministic.pdf(np.arange(-0.1, 0.1, 0.01))
# The computation of unspecified common methods can become very slow, since only general methods are called
# which, by their very nature, cannot use any specific information about the distribution.
# Thus, as a cautionary example:
from scipy.integrate import quad
quad(deterministic.pdf, -1e-1, 1e-1)
# But this is not correct: the integral over this pdf should be 1.
# Let’s make the integration interval smaller:
quad(deterministic.pdf, -1e-3, 1e-3)
# This looks better. However, the problem originated from the fact that
# the pdf is not specified in the class definition of the deterministic distribution.

# Subclassing rv_discrete
# a discrete distribution that has the probabilities of the truncated normal
# for the intervals centered around the integers
'''You can construct an arbitrary discrete rv where P{X=xk} = pk
    by passing to the rv_discrete initialization method (through the values= keyword)
    a tuple of sequences (xk, pk) which describes only those values of X (xk)
    that occur with nonzero probability (pk).”
  The keyword name is required.
  The support points of the distribution xk have to be integers.
  The number of significant digits (decimals) needs to be specified.'''
# example
npoints = 20   # number of integer support points of the distribution minus 1
npointsh = npoints / 2
npointsf = float(npoints)
nbound = 4   # bounds for the truncated normal
normbound = (1+1/npointsf) * nbound   # actual bounds of truncated normal
grid = np.arange(-npointsh, npointsh+2, 1)   # integer grid
gridlimitsnorm = (grid-0.5) / npointsh * nbound   # bin limits for the truncnorm
gridlimits = grid - 0.5   # used later in the analysis
grid = grid[:-1] # drop the last element
probs = np.diff(stats.truncnorm.cdf(gridlimitsnorm, -normbound, normbound))
gridint = grid
# sublassing rw_discrete (isn't it rather a freezing of a distribution?)
normdiscrete = stats.rv_discrete(
  values=(gridint,
          np.round(probs, decimals=7)),
  name='normdiscrete')
# access to all common methods of discrete distributions
print 'mean = %6.4f, variance = %6.4f, skew = %6.4f, kurtosis = %6.4f'% \
      normdiscrete.stats(moments =  'mvsk')
nd_std = np.sqrt(normdiscrete.stats(moments='v'))
nd_std

# Testing the Implementation

# generate a random sample and compare observed frequencies with the probabilities
n_sample = 500
np.random.seed(87655678)   # fix the seed for replicability
rvs = normdiscrete.rvs(size=n_sample)
rvsnd = rvs
f, l = np.histogram(rvs, bins=gridlimits)
sfreq = np.vstack([gridint, f, probs*n_sample]).T
print sfreq

''' PLOT '''
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#
npoints = 20 # number of integer support points of the distribution minus 1
npointsh = npoints / 2
npointsf = float(npoints)
nbound = 4 #bounds for the truncated normal
normbound = (1 + 1 / npointsf) * nbound #actual bounds of truncated normal
grid = np.arange(-npointsh, npointsh+2, 1) #integer grid
gridlimitsnorm = (grid-0.5) / npointsh * nbound #bin limits for the truncnorm
gridlimits = grid - 0.5
grid = grid[:-1]
probs = np.diff(stats.truncnorm.cdf(gridlimitsnorm, -normbound, normbound))
gridint = grid
normdiscrete = stats.rv_discrete(
                        values=(gridint, np.round(probs, decimals=7)),
                        name='normdiscrete')
#
n_sample = 500
np.random.seed(87655678) #fix the seed for replicability
rvs = normdiscrete.rvs(size=n_sample)
rvsnd=rvs
f,l = np.histogram(rvs, bins=gridlimits)
sfreq = np.vstack([gridint, f, probs*n_sample]).T
fs = sfreq[:,1] / float(n_sample)
ft = sfreq[:,2] / float(n_sample)
nd_std = np.sqrt(normdiscrete.stats(moments='v'))
#
ind = gridint  # the x locations for the groups
width = 0.35       # the width of the bars
#
plt.subplot(111)
rects1 = plt.bar(ind, ft, width, color='b')
rects2 = plt.bar(ind+width, fs, width, color='r')
normline = plt.plot(ind+width/2.0, stats.norm.pdf(ind, scale=nd_std),
                    color='b')
#
plt.ylabel('Frequency')
plt.title('Frequency and Probability of normdiscrete')
plt.xticks(ind+width, ind)
plt.legend((rects1[0], rects2[0]), ('true', 'sample'))
#
plt.show()
''' end of plot '''
''' PLOT '''
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#
npoints = 20 # number of integer support points of the distribution minus 1
npointsh = npoints / 2
npointsf = float(npoints)
nbound = 4 #bounds for the truncated normal
normbound = (1 + 1 / npointsf) * nbound #actual bounds of truncated normal
grid = np.arange(-npointsh, npointsh+2,1) #integer grid
gridlimitsnorm = (grid - 0.5) / npointsh * nbound #bin limits for the truncnorm
gridlimits = grid - 0.5
grid = grid[:-1]
probs = np.diff(stats.truncnorm.cdf(gridlimitsnorm, -normbound, normbound))
gridint = grid
normdiscrete = stats.rv_discrete(
                        values=(gridint, np.round(probs, decimals=7)),
                        name='normdiscrete')
#
n_sample = 500
np.random.seed(87655678) #fix the seed for replicability
rvs = normdiscrete.rvs(size=n_sample)
rvsnd = rvs
f,l = np.histogram(rvs,bins=gridlimits)
sfreq = np.vstack([gridint,f,probs*n_sample]).T
fs = sfreq[:,1] / float(n_sample)
ft = sfreq[:,2] / float(n_sample)
fs = sfreq[:,1].cumsum() / float(n_sample)
ft = sfreq[:,2].cumsum() / float(n_sample)
nd_std = np.sqrt(normdiscrete.stats(moments='v'))
#
ind = gridint  # the x locations for the groups
width = 0.35   # the width of the bars
#
plt.figure()
plt.subplot(111)
rects1 = plt.bar(ind, ft, width, color='b')
rects2 = plt.bar(ind+width, fs, width, color='r')
normline = plt.plot(ind+width/2.0, stats.norm.cdf(ind+0.5,scale=nd_std),
                    color='b')
#
plt.ylabel('cdf')
plt.title('Cumulative Frequency and CDF of normdiscrete')
plt.xticks(ind+width, ind)
plt.legend((rects1[0], rects2[0]), ('true', 'sample'))
#
plt.show()
''' end of plot '''
# test, whether our sample was generated by our normdiscrete distribution.
# This also verifies whether the random numbers are generated correctly
# The chisquare test requires that there are a minimum number of observations in each bin.
# We combine the tail bins into larger bins so that they contain enough observations.
f2 = np.hstack([f[:5].sum(), f[5:-5], f[-5:].sum()])
p2 = np.hstack([probs[:5].sum(), probs[5:-5], probs[-5:].sum()])
ch2, pval = stats.chisquare(f2, p2*n_sample)
print 'chisquare for normdiscrete: chi2 = %6.3f pvalue = %6.4f' % (ch2, pval)
# the pvalue in this case is high, so we can be quite confident that our random sample was actually generated by the distribution

# Analysing One Sample
# rvs
np.random.seed(282629734)
x = stats.t.rvs(10, size=1000)
# Descriptive Statistics
print x.max(), x.min()  # equivalent to np.max(x), np.min(x)
print x.mean(), x.var() # equivalent to np.mean(x), np.var(x)
m, v, s, k = stats.t.stats(10, moments='mvsk') # theoretical properties
n, (smin, smax), sm, sv, ss, sk = stats.describe(x) # sample properties
  # stats.describe uses the unbiased estimator for the variance, while np.var is the biased estimator
print 'distribution:',
sstr = 'mean = %6.4f, variance = %6.4f, skew = %6.4f, kurtosis = %6.4f'
print sstr %(m, v, s ,k)
print 'sample:      ',
print sstr %(sm, sv, ss, sk)
# T-test and KS-test

















