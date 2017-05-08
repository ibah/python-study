# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:56:17 2015

@author: msiwek
"""

import numpy as np

''' Basics '''

''' example '''
# reshape(ncol) or nrow???
# reshape(nrow, ncol)
# reshape(nslice, nrow, ncol)
a = np.arange(15).reshape((3, 5))
a
a = np.arange(15).reshape(3, 5)  # the same
a
a.shape
a.ndim
a.dtype.name
a.itemsize
a.size
type(a)
b = np.array([6, 7, 8])
b
type(b)

np.arange(10)
np.array([[1, 2, 3], [4, 5, 6]])
np.array([[1, 2, 3], [4, 5]])
np.array([[1, 2, 3], 4])
a = np.arange(15).reshape((3, 5), order='C')  # default (c like)
a[1,2]  # a[row, col]
a = np.arange(15).reshape((3, 5), order='F')  # fortran like
a[1,2]  # the same
a = np.arange(60).reshape(3, 4, 5)
a[1,2,3] # a[slice, row, col]

''' Array Creation '''

# 1 (from a list)
a = np.array([2,3,4])
a
a.dtype
b = np.array([1.2, 3.5, 5.1])
b.dtype

# 2 (from a tuple)
b = np.array([(1.5,2,3),(4,5,6)]) # seq of seq creates a 2-dim array
b

# 3 explicitly specify the type
c = np.array( [ [1,2], [3,4] ], dtype=complex )
c
# 3 empty arrays
# zeros creates an array full of zeros
# ones creates an array full of ones
# empty creates an array whose initial content is random
  # and depends on the state of the memory
# By default, the dtype of the created array is float64.
np.zeros( (3,4) )
np.ones( (2,3,4), dtype=np.int16 )                # dtype can also be specified
np.empty( (2,3) )

# 4 sequences of numbers
np.arange( 10, 30, 5 ) # by step
np.arange( 0, 2, 0.3 )                 # it accepts float arguments
# When arange is used with floating point arguments:
# it is usually better to use the function linspace that:
  # the number of elements that we want, instead of the step.
np.linspace( 0, 2, 9 )                 # 9 numbers from 0 to 2
x = np.linspace( 0, 2*np.pi, 100 )        # useful to evaluate function at lots of points
f = np.sin(x)
import matplotlib.pyplot as plt
plt.plot(f)

# create based on a given array
np.zeros_like
np.ones_like
np.empty_like
# fill in with randoms:
np.random.rand  # uniform
np.random.randn  # normal
# fill in according to another source
np.fromfunction  # using a custom function
np.fromfile  # using a file or string
# example:
def f(x,y): # this is vectorized
    print(x)
    print(y)
    return 10*x+y
nrow = 2
ncol = 3
np.fromfunction(f, (nrow, ncol), dtype=int)

''' Printing Arrays '''

np.arange(6)                         # 1d array
np.arange(12).reshape(4,3)           # 2d array
np.arange(24).reshape(2,3,4)         # 3d array
print np.arange(10000)
print np.arange(10000).reshape(100,100)
# set_printoptions(threshold='nan') # forcing to print whole arrays

''' Basic Operations '''

# Arithmetic operators on arrays apply elementwise.
# A new array is created and filled with the result.
a = np.array( [20,30,40,50] )
b = np.arange( 4 )
b
c = a-b
c
b**2
10*np.sin(a)
a<35

# the product operator * operates elementwise in NumPy arrays
# The matrix product can be performed using:
  #  the dot function or
  # creating matrix objects ( see matrix section of this tutorial )
A = np.array( [[1,1], [0,1]] )
B = np.array( [[2,0], [3,4]] )
A*B                         # elementwise product
np.dot(A,B)                    # matrix product

# += and *= modify an existing array
a = np.ones((2,3), dtype=int)
b = np.random.random((2,3))    # rand[0-1] in 2 rows and 3 cols
    # random takes 1 argument being a tuple giving the dims of an array
    # if not arg, then a single float returned
a *= 3
a
b += a
b
a += b                      # b is converted to integer type
a                           # info is being lost !!!!!!!!!!!!!!!!!!!!

# upcasting: the resulting array corresponds to the more general or precise
a = np.ones(3, dtype=np.int32)
b = np.linspace(0,np.pi,3)
a
b
b.dtype
b.dtype.name
c = a+b
c
c.dtype.name
c*1j
d = np.exp(c*1j)   # 1*j
d
d.dtype.name

# Many unary operation -> methods of the ndarray class
# they ignore the shape, treat the data as a list of numbers
a = np.random.random((2,3))
a
a.sum()
a.min()
a.max()
# apply an operation along the specified axis of an array:
b = np.arange(12).reshape(3,4)
b
b.sum(axis=0)  # sum along 0 = sum of each column
b.min(axis=1)  # min along 1 = min of each row
b.cumsum(axis=1)  # cumulative sum along each row

''' Universal Functions (ufunc), elementwise'''

B = np.arange(3)
B
np.exp(B)
np.sqrt(B)
C = np.array([2., -1., 4.])
np.add(B, C)  # element-wise addition
# see more functions:
# all, alltrue, any, apply_along_axis, argmax, argmin, argsort, average,
# bincount, ceil, clip, conj, conjugate, corrcoef, cov, cross, cumprod, cumsum,
# diff, dot, floor, inner, inv, lexsort, max, maximum, mean, median, min,
# minimum, nonzero, outer, prod, re, round, sometrue, sort, std, sum, trace,
# transpose, var, vdot, vectorize, where 

''' Indexing, Slicing and Iterating '''

# b[ncol], b[nrow, ncol], b[nslice, nrow, ncol]
# One-dimensional arrays can be indexed, sliced and iterated over like a list
a = np.arange(10)**3
a
a[2:5]
a[:6:2] = -1000 # equivalent to a[0:6:2] = -1000; from start to position 6,
                # exclusive, set every 2nd element to -1000
a
a[::-1]  # reversed order
a[-2:-1] # the two last elements
for i in a:
    print i**(1/3.)  # the dot is necessary to convert 3 into float
                     # and so to have a normal float division 1/3
a = np.arange(10)

a[0]      # element 1 = 0
a[0:1]    # array consisting of element 1 only = [0]
a[:1]     # ditto

a[-1]     # the element 10 (the last) = 9
a[9]      # ditto
a[-1:]    # array consisting of element 10 only = [9]

a[-2]     # the element 9 = 8
a[-2:-1]  # array consisiting of the element 9 only = [8]
a[-2:0]   # empty array (elements form 9 to 1, by +1, i.e. null)
a[-2:]    # array consisting of elements 8-9 (the last two) = [8,9]

# Multidimensional arrays can have one index per axis
# a tuple separated by commas
def f(x,y):
    return 10*x+y
# Construct an array by executing a function over each coordinate
b = np.fromfunction(f,(5,4),dtype=int)  # 5 rows, 4 columns
b
b[2,3]  # row 2 (third), col 3 (forth)
b[0:5, 1]  # each row in the second column of b
b[ : ,1]   # equivalent to the previous example
b[1:3, : ] # each column in the second and third row of b

# missing index is a full slice
# [1-dim output] getting a list
b[-1]     # the last row. Equivalent to b[-1,:], here a list of numbers
b[-1,:]   # the same
b[-1,...] # the same
b[0]
# [2-dim output] getting a matrix
b[-1:]      # the last row but here a list of list of numbers
b[-1:,...]  # the same
b[0:1]
b[0:2]
# list vs. matrix output
b1 = b[-1]
b2 = b[-1:]
b1[1]
b2[1]  # error
b2[0, 1]  # now correct
# Summary:
# if the index is a number -> the corresponding dimension is dropped
b       # 2-dim
b[0]    # 1-dim
b[0,0]  # 0-dim
# if the index is a sequence (even 1-element sequence) -> the dimension is maintained
b[0:1]  # 2-dim
b[0:1,0:1] # 2-dim

# dots (...)
# represent as many colons as needed to produce a complete indexing tuple.
c = np.array( [[[  0,  1,  2],             # a 3D array (two stacked 2D arrays)
                [ 10, 12, 13]],
               [[100,101,102],
                [110,112,113]]])
c.shape
c[1,...]                                   # same as c[1,:,:] or c[1]
c[...,2]                                   # same as c[:,:,2]
# Iterating
# over multidimensional arrays is done with respect to the first axis
b
for row in b:
  print(row)
# flat = iterator over all elements of the matrix
for element in b.flat:
  print(element)

# See also:
# Indexing, newaxis, ndenumerate, indices

''' Shape Manipulation '''

''' Changing the shape of an array '''

a = np.floor(10*np.random.random((3,4)))
a
a.shape
# shape changing (not affecting)
a.ravel()         # flatten the array (not affecting the array itself)
a
a.T               # transformation (not affecting the array itself)
a
a.reshape(6,2)    # reshaping (not affecting the array itself)
a
a.reshape(4,-1)   # -1 autmatically calculates the missing dimensions
a
# changing shape of the array
a.shape = (3, 4)  # resizing using 'shape' - affects the array itself
a
a.resize((2,6))   # resizing - affects the array itself
a

# See also: ndarray.shape, reshape, resize, ravel

''' Stacking together different arrays '''

a = np.floor(10*np.random.random((2,2)))
a
b = np.floor(10*np.random.random((2,2)))
b
np.vstack((a,b))
np.hstack((a,b))
# The function column_stack stacks 1D arrays as columns into a 2D array.
# It is equivalent to vstack only for 1D arrays:
from numpy import newaxis
np.column_stack((a,b))   # With 2D arrays
a = np.array([4.,2.])
b = np.array([2.,8.])
a[:,newaxis]  # This allows to have a 2D columns vector
np.column_stack((a[:,newaxis],b[:,newaxis]))
np.vstack((a[:,newaxis],b[:,newaxis])) # The behavior of vstack is different

# For arrays of with more than two dimensions:
  # hstack stacks along their second axes
  # vstack stacks along their first axes
  # concatenate allows for an optional arguments giving
    # the number of the axis along which the concatenation should happen.

# In complex cases, r_ and c_ are useful for creating arrays by stacking numbers along one axis.
  # They allow the use of range literals (”:”)
np.r_[1:4,0,4]

''' Splitting one array into several smaller ones '''

#  hsplit = split an array along its horizontal axis, either
  # by specifying the number of equally shaped arrays to return, or
  # by specifying the columns after which the division should occur
# vsplit = split along the vertical axis
# array_split = allows one to specify along which axis to split.
a = np.floor(10*np.random.random((2,12)))
a
np.hsplit(a,3)   # Split a into 3
  # imagine: split a table into 3 subtables
    # each having all rows but only a subset of adjacent columns
np.hsplit(a,(3,4))   # Split a after the third and the fourth column
  # here: tables having cols [0,1,2], [3], [4,5...,11]

''' Copies and Views '''

  # When operating and manipulating arrays
  # their data is sometimes copied into a new array and sometimes not.
# There are three cases:
# 1. No Copy at All
  # Simple assignments
a = np.arange(12)
b = a            # no new object is created
b is a           # a and b are two names for the same ndarray object
b.shape = 3,4    # changes the shape of a
a.shape
  # Python passes mutable objects as references, so function calls make no copy
def f(x):
    print(id(x))
id(a)                           # id is a unique identifier of an object
f(a)
# 2. View or Shallow Copy
  # Different array objects can share the same data.
  # The view method creates a new array object that looks at the same data.
c = a.view()
c is a
c.base is a                        # c is a view of the data owned by a
c.flags.owndata
c.shape = 2,6                      # a's shape doesn't change
a.shape
c[0,4] = 1234                      # a's data changes
a
  # Slicing an array returns a view of it
s = a[ : , 1:3]     # spaces added for clarity; could also be written "s = a[:,1:3]"
    #rows,cols = [all = 0,1,2; 1:3 = 1,2]
s
s[:] = 10           # s[:] is a view of s. Note the difference between s=10 and s[:]=10
    # s = 10, s is an int scalar
    # s[:] = 10, assigns 10 to all elements of s
a
# 3. Deep Copy
  # The copy method makes a complete copy of the array and its data.
d = a.copy()                          # a new array object with new data is created
d is a
d.base is a                           # d doesn't share anything with a
d[0,0] = 9999
a
d

''' Functions and Methods Overview '''
# see ....

''' Broadcasting rules '''
# allow universal functions to deal in a meaningful way
  # with inputs that do not have exactly the same shape
# 1
  # if all input arrays do not have the same number of dimensions,
  # a “1” will be repeatedly prepended to the shapes of the smaller arrays
  # until all the arrays have the same number of dimensions
# 2
  # arrays with a size of 1 along a particular dimension act as if
  # they had the size of the array with the largest shape along that dimension.
  # The value of the array element is assumed to be the same along that dimension for the “broadcast” array
# result: the sizes of all arrays must match

''' Fancy indexing and index tricks '''
# In addition to indexing by:
  # integers and
  # slices, as we saw before
# arrays can be indexed by:
  # arrays of integers and
  # arrays of booleans

''' Indexing with Arrays of Indices '''
# for 1-dim array:
a = np.arange(12)**2                       # the first 12 square numbers
i = np.array( [ 1,1,3,8,5 ] )              # an array of indices
a[i]                                       # the elements of a at the positions i
j = np.array( [ [ 3, 4], [ 9, 7 ] ] )      # a bidimensional array of indices
a[j]                                       # the same shape as j

# for multi-dim array:
  # when index array is multi-dim then a single array of indices refers to the first dimension
palette = np.array( [ [0,0,0],                # black
                      [255,0,0],              # red
                      [0,255,0],              # green
                      [0,0,255],              # blue
                      [255,255,255] ] )       # white
image = np.array( [ [ 0, 1, 2, 0 ],           # each value corresponds to a color in the palette
                    [ 0, 3, 4, 0 ]  ] )
palette[image]                            # the (2,4,3) color image
  # multi-dim index arrays  
  # The arrays of indices for each dimension must have the same shape
a = np.arange(12).reshape(3,4)
a
i = np.array( [ [0,1],                        # indices for the first dim of a
                [1,2] ] )
j = np.array( [ [2,1],                        # indices for the second dim
                [3,3] ] )
a[i,j]                                     # i and j must have equal shape
a[i,2]
a[:,j]

# indexing with a sequence (list, tuple) vs. with an array
  # we can put i and j in a sequence (list or tuple) and then do the indexing with the list
l = [i,j]
m = (i,j)
a[l]                                       # equivalent to a[i,j]
a[m]
  # we can not do this by putting i and j into an array,
  # because this array will be interpreted as indexing the first dimension of a.
s = np.array( [i,j] )
a[s]                                       # not what we want
a[tuple(s)]                                # same as a[i,j]

# the search of the maximum value of time-dependent series
time = np.linspace(20, 145, 5)                 # time scale
  # time moves along the row (column by column) (5 moments)
data = np.sin(np.arange(20)).reshape(5,4)      # 4 time-dependent series
  # time moves along columns: row by row (5 moments = 5 rows)
  # each column is a separate time series (so 4 time series)
time
data
ind = data.argmax(axis=0)                   # index of the maxima for each series
  # for each column (time series) find the maximum value and report its index (row index)
  # so this is:
    # apply along 0 axis = along rows
    # -> rows will collaps, columns will be preserved
ind
time_max = time[ ind]                       # times corresponding to the maxima
  # replace indeces with time values (moments)
data_max = data[ind, xrange(data.shape[1])] # => data[ind[0],0], data[ind[1],1]...
  # xrange is the lazy range (i.e. object yielding indeces on demand)
  # here it's applied for the 1 axis i.e. for columns
  # so it will yield 0,1,2,3
    # data[row_index_of_max_value_in_col_0, 0 yielded,
    #      row_index_of_max_value_in_col_1, 1 yielded,
    #      ...,
    #      row_index_of_max_value_in_col_3, 3 yielded]
time_max
data_max
np.all(data_max == data.max(axis=0))  # are all elements True?
  # comparing values retrived by first obtaining the indeces
  # with values pulled directly by asking for maxima
  # np.max() function:
    # if no axis provided then flatten array (global maximum)
    # if axis=0 then axis 0 collapses (rows collapse)
      # i.e. maxium along rows for each column is provided

# indexing with arrays as a target to assign to
a = np.arange(5)
a
a[[1,3,4]] = 0
a
# when the list of indices contains repetitions,
  # the assignment is done several times, leaving behind the last value
a = np.arange(5)
a[[0,0,2]]=[1,2,3]  # element 0 is assign 1 and 2; 2 is left
a
# be careful with +=
a = np.arange(5)
a[[0,0,2]]+=1 # here +1 is done only once for element 0
a

''' Indexing with Boolean Arrays '''
# boolean arrays that have the same shape as the original array
a = np.arange(12).reshape(3,4)
b = a > 4
b    # b is a boolean with a's shape
a[b] # 1d array with the selected elements
    # this is just a flat array of selected elements
# useful in assignment
a[b] = 0                                   # All elements of 'a' higher than 4 become 0
a
# how to use boolean indexing to generate an image of the Mandelbrot set:
import numpy as np
import matplotlib.pyplot as plt
def mandelbrot( h,w, maxit=20 ):
    """Returns an image of the Mandelbrot fractal of size (h,w)."""
    y,x = np.ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
    c = x+y*1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2            # who is diverging
        div_now = diverge & (divtime==maxit)  # who is diverging now
        divtime[div_now] = i                  # note when
        z[diverge] = 2                        # avoid diverging too much

    return divtime
plt.imshow(mandelbrot(400,400))
plt.show()

# for each dimension of the array we give a 1D boolean array selecting the slices we want
a = np.arange(12).reshape(3,4)
b1 = np.array([False,True,True])             # first dim selection
b2 = np.array([True,False,True,False])       # second dim selection
a[b1,:]                                   # selecting rows
a[b1]                                     # same thing
a[:,b2]                                   # selecting columns
a[b1,b2]                                  # a weird thing to do

''' The ix_() function '''
# The ix_ function can be used to combine different vectors so as to obtain the result for each n-uplet
a = np.array([2,3,4,5])
b = np.array([8,5,4])
c = np.array([5,4,6,8,3])
ax,bx,cx = np.ix_(a,b,c)
ax
bx
cx
ax.shape, bx.shape, cx.shape
  # (slices, rows, cols)
result = ax+bx*cx
result
result[3,2,4]
  # [slice, row, col]
a[3]+b[2]*c[4]

# You could also implement & use the reduce as follows
def ufunc_reduce(ufct, *vectors):
   vs = np.ix_(*vectors)
   r = ufct.identity
   for v in vs:
       r = ufct(r,v)
   return r
ufunc_reduce(np.add,a,b,c)
# The advantage of this version of reduce compared to the normal ufunc.reduce is that
# it makes use of the Broadcasting Rules in order to avoid creating an argument array
# the size of the output times the number of vectors

''' Indexing with strings '''
# see RecordArray

''' Linear Algebra '''

# Simple Array Operations
  # See linalg.py in numpy folder
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
print(a)
a.transpose()
np.linalg.inv(a)
u = np.eye(2) # unit 2x2 matrix; "eye" represents "I"
u
j = np.array([[0.0, -1.0], [1.0, 0.0]])
j
np.dot (j, j) # matrix product
np.trace(u)  # trace
y = np.array([[5.], [7.]])
np.linalg.solve(a, y)
np.linalg.eig(j)

''' Tricks and Tips '''

# “Automatic” Reshaping
# omit one of the sizes
a = np.arange(30)
a.shape = 2,-1,3  # -1 means "whatever is needed"
a.shape
a

# Vector Stacking
# the functions column_stack, dstack, hstack and vstack
# depending on the dimension in which the stacking is to be done
x = np.arange(0,10,2)
y = np.arange(5)
x
y
mv = np.vstack([x,y])
mh = np.hstack([x,y])
mv
mh

# histogram
# The NumPy histogram function applied to an array returns a pair of vectors:
  # the histogram of the array and the vector of bins.
# Comparison with hist:
  # (matplotlib) pylab.hist plots the histogram automatically
  # numpy.histogram only generates the data.

import numpy as np
import matplotlib.pyplot as plt
# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
mu, sigma = 2, 0.5
v = np.random.normal(mu,sigma,10000)

# Plot a normalized histogram with 50 bins
plt.hist(v, bins=50, normed=1)       # matplotlib version (plot)
plt.show()

# Compute the histogram with numpy and then plot it
(n, bins) = np.histogram(v, bins=50, normed=True)  # NumPy version (no plot)
plt.plot(.5*(bins[1:]+bins[:-1]), n)
plt.show()


''' More '''


np.genfromtxt() # reading data from file
np.meshgrid() # Return coordinate matrices from two or more coordinate vectors.
np.random.choice() # Generates a random sample from a given 1-D array
np.put() # Replaces specified elements of an array with given values.
  # The indexing works on the flattened target array. put is roughly equivalent to:
  # a.flat[ind] = v
np.logical_not  # vectorized versions of the logical operators
























