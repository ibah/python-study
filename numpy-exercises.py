# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:44:05 2016

@author: msiwek
"""

''' Timing

%timeit expression

import timeit
timeit.Timer(statement, setup)
.timeit()
.repeat()

'''

# 1. Import the numpy package under the name np
import numpy as np
# Print the numpy version and the configuration
np.version.version
np.__version__
print np.__version__
np.__config__.show()
print np.version  # wrong: this is just a module
# Create a null vector of size 10
np.zeros(10)
# How to get the documentation of the numpy add function from the command line?
'''python -c "import numpy; numpy.info(numpy.add)"  # command line
np.add?  # interactive mode'''
help(np.add)  # withing a script
np.add  # hover the mouse pointer and press ctrl+i
np.info(np.add)  # doesn't work, I don't know why
# Create a null vector of size 10 but the fifth value which is 1
a = np.zeros(10)
a[4] = 1
a
# Reverse a vector (first element becomes last) 
a = np.arange(10)
a[::-1]
# Create a vector with values ranging from 10 to 49
np.arange(10,50)
# Create a 3x3 matrix with values ranging from 0 to 8
np.arange(9).reshape(3, 3)
# Find indices of non-zero elements from [1,2,0,0,4,0]
a = [1,2,0,0,4,0]
np.nonzero(a)
# 10. Create a 3x3 identity matrix
np.eye(3)
# Create a 3x3x3 array with random values 
np.random.rand(3,3,3)
np.random.random((3,3,3)) # here: scientific/exponential notation
np.random.random(27).reshape(3,3,3)
# Create a 10x10 array with random values and find the minimum and maximum values
a = np.random.rand(10,10)
a.min()
a.max()
# Create a random vector of size 30 and find the mean value
np.random.rand(30).mean()
# How to print all the values of an array ?
  # official
#np.set_printoptions(threshold=np.nan)
#Z = np.zeros((25,25))
#print(Z)
# Create a 2d array with 1 on the border and 0 inside
  # 1
a = np.zeros((10,10),dtype=int)
a[0,]=1
a[9,]=1
a[:,0]=1
a[:,9]=1
a
  # official
Z = np.ones((10,10))
Z[1:-1,1:-1] = 0
Z
  # check
a = np.arange(100).reshape(10,10)
a
a[1:-1,] # all except the first and the last row
a[:,1:-1] # all except the first and the last col
  # 2
a = np.zeros((10,10),dtype=int)
a[[0,9],]=1
a[:,[0,9]]=1
a
  # 3
a = np.ones((10,10),dtype=int)
a[1:-1,1:-1]=0
a

# What is the result of the following expression?
0 * np.nan # nan
np.nan == np.nan # False
np.inf > np.nan # False
np.nan - np.nan # nan
0.3 == 3 * 0.1 # False, as RHS = 0.30000000000000004

# Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
np.diag(np.arange(5))
np.diag(np.arange(5), 1)
np.diag(np.arange(5), 2)
  # 1
np.diag(np.arange(1,5), -1)
np.diag(np.arange(4)+1, -1)
  # 2
a = np.zeros((5,5), dtype=np.int)  # default is numpy.float64
for i in np.arange(1,5):
  a[i,i-1] = i
a
  # 3, best
np.diag(np.arange(1,5),-1)

# Create a 8x8 matrix and fill it with a checkerboard pattern
  # 0, best
a = np.ones((8,8),dtype=np.int)
a[::2,::2]=0
a[1::2,1::2]=0
a
  # 1
def f(a, b):
  return np.mod(a+b,2)
a = np.fromfunction(f, (8,8), dtype=int)
a
  # 2
a = np.zeros((8,8), dtype=int)
a = np.arange(64).reshape(8,8)
a
a[::2]       # every second row (even: 0, 2, ...)
a[1::2]      # every second row (odd: 1, 3, ...)
a[...,::2]   # every second column (even)
a[...,1::2]  # every second column (odd)
a[::2,1::2]  # every odd column in every even row
a[1::2,::2]  # every even column in every odd row
a = np.zeros((8,8), dtype=int)
a[::2,1::2] = 1  # filling in even rows
a[1::2,::2] = 1  # filling in odd rows
a
  # 3 wrong
np.vstack((np.zeros(32, dtype=int), np.ones(32, dtype=int))).T.reshape(8,8)
  # official
Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)

# Create a checkerboard 8x8 matrix using the tile function
  # checking
a = np.arange(2)
np.tile(a,[8,4]) # no, this is without pattern shifting
np.mod(np.arange(4).reshape(2,2),2) # wrong again
  # 0, ok
np.tile(np.array((0,1,1,0)).reshape(2,2),(4,4))
  # checking
a = np.array([0, 1, 2])
np.tile(a, 2)
np.tile(a, (2, 2))
np.tile(a, (2, 1, 2))
  # 1, wrong
np.tile(np.diag(np.ones(2, dtype=int)),(4,4))  # almost
  # 2, best
np.tile(np.array([[0,1],[1,0]]),(4,4))  # correct
np.tile(((0,1),(1,0)),(4,4))  # shortest

# Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element
a = np.arange(6*7*8).reshape(6,7,8)
a
np.ravel(a)
np.unravel_index(0,(1,1))
  # official
np.unravel_index(100,(6,7,8))

# Normalize a 5x5 random matrix

a = np.random.random((5,5))
  # 1, mean=0, std=1
mn = np.mean(a)
std = np.std(a)
b = (a - mn) / std
b
np.mean(b)
np.std(b)
  # 2, official, [min,max] = [0,1]
amax, amin = a.max(), a.min()
(a - amin)/(amax - amin)
  # 3 (same as 1)
a = np.random.rand(5,5)
a = np.random.random((5,5))
a.mean()
np.mean(a)
np.std(a)
(a - np.mean(a))/np.std(a)

# Create a custom dtype that describes a color as four unisgned bytes (RGBA)
  # official
color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])
color
  # checking
np.dtype([('f1', [('f1', np.int16)])])
np.dtype([('a','f8'),('b','S10')])
np.dtype("i4, (2,3)f8")
np.dtype([('hello',(np.int,3)),('world',np.void,10)])
np.dtype((np.int16, {'x':(np.int8,0), 'y':(np.int8,1)}))
np.dtype({'names':['gender','age'], 'formats':['S1',np.uint8]})

# 18. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) 
a = np.random.random((5,3))
b = np.random.random((3,2))
a,b
np.dot(a,b)

# Given a 1D array, negate all elements which are between 3 and 8, in place.
a = np.random.randint(10, size = 20)
a
(a>=3)&(a<=8)
  # 1, using np.select()
condlist = [(a>=3)&(a<=8),a<3,a>8]
choicelist = [-a,a,a]
np.select(condlist,choicelist)
  # 2, using np.where()
np.where((a>=3)&(a<=8),-a,a)
  # 3, using np.choose()
B = (a>=3)&(a<=8).astype(int)
choices = [a,-a]
np.choose(B,choices)
B.choose(choices)
  # 4, official, using multiplication in place - the only one working in place
Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1
  # checking
a = np.arange(10)
a = 1 # turns a into a scalar
a[a<6] = 1 # subistitutes only selected positions
a[:] = 1 # substitutes all positions in a
a *= 1 # no effect
a[a<6] *= 1 # no effect
a[:] *= 1 # no effect

# What is the output of the following script
sum(range(5),-1)
  # here it is sum(sequence, start=0) = sum(sequence) + start
  # from numpy import *
np.sum(range(5),-1)
  # here it is sum(sequence, axis)
  # checking
a = np.arange(100).reshape(10,10)
np.sum(a,0) # sum along rows, so sum of col 0, sum of col 1, ...
np.sum(a,1) # sum along cols, so sum of row 0, sum of row 1, ...
np.sum(a,3) # axis out of bounds
np.sum(a,-1) # sum along the last axis, i.e. along cols, so sum of row 0, ...
np.sum(a,-2) # sum along the last but one axis, i.e. along rows, so sum of col 0, ...
np.sum(a,-3) # asis out of bounds

# Consider an integer vector Z, which of these expressions are legal?
Z = np.arange(10)
Z
Z**Z  # OK, element-wise exponentiation
2 << Z # binary left shift: 10 is shifted to left: 10, 100, 1000, ...
    # 2, 4, 8, ...
Z << 2 # binary left shift: Z is shifted by 2: 0, 100, 1000, 1100, 10000
    # 0, 4, 8, 12, 16
Z >> 2 # binary right shift: Z is shifted by 2:
    # 0->0, 1->0. 10->0, 11->0, 100->1, ..., 1000-> 10,
2 << Z >> 2 # shift left and right, no change
Z <- Z  # lower than -Z
1j*Z  # turn into imaginary
Z/1/1 # div by 1 and by 1, no change except for type conversion implied by "/"
Z<Z # element-wise comparison
Z>Z
Z<Z>Z # error

# What are the result of the following expressions
np.array(0) // np.array(0) # floor division = 0
np.array(0) // np.array(0.) # all else is nan
np.array(0) / np.array(0)
np.array(0) / np.array(0.)

# How to round away from zero a float array
a = (10 - (-10))*np.random.random(10) - 10
a
np.trunc(a) # towards 0
np.copysign(1,a) # array of 1-s with the same signs as elements of a
    # official - wrong
Z = np.random.uniform(-10,+10,10)
Z = np.array([1.4])
print (np.trunc(Z + np.copysign(0.5, Z)))
    # correct
Z = -Z
np.copysign(np.ceil(np.abs(Z)), Z)
    # another - wrong as well
Z = np.array([1.,1.5,2.0,2.5,3.0,3.5])
Z = Z
print (np.round(Z + np.copysign(0.5, Z)))

# Extract the integer part of a random array using 5 different methods
    #np.random.random(10)
Z = np.random.uniform(0,10,10)
    # sic: all the solutions can give different results for negative floats
print (Z - Z%1) # np.mod(Z,1)
print (np.floor(Z))
print (np.ceil(Z)-1) # wrong for integers
print (Z.astype(int))
print (np.trunc(Z))

# Create a 5x5 matrix with row values ranging from 0 to 4
    # 1, this is row-wise
np.repeat(np.arange(5),5).reshape(5,5)
    # 2, this is col-wise
np.tile(np.arange(5),5).reshape(5,5)
    # 3, official - this is col-wise
Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)
    # 4, official, shortened
np.zeros((5,5), dtype=int) + np.arange(5) # col-wise
np.zeros((5,5), dtype=int) + np.expand_dims(np.arange(5),1) # row-wise
np.zeros((5,5), dtype=int) + np.arange(5)[:,np.newaxis] # row-wise

# Consider a generator function that generates 10 integers and use it to build an array
def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float)
print(Z)

# Create a vector of size 10 with values ranging from 0 to 1, both excluded
    # 1
np.linspace(0,1,10)
np.arange(0,1,0.1)
np.linspace(0,0.9,10)
  # correct
np.linspace(0,1,12,endpoint=True)[1:-1]

# Create a random vector of size 10 and sort it
np.sort(np.random.random(10))

# How to sum a small array faster than np.sum? 
    # official
Z = np.arange(10)
np.add.reduce(Z)  # equivalent to np.sum()

# Consider two random array A anb B, check if they are equal
A = np.random.randint(5, size = 3)
B = np.random.randint(5, size = 3)
B = A
B = np.array([[10,10],[20,20]])
B = np.array([])
A == B
np.all(A == B)
(A==B).all()
# checking
    np.array([1])==np.array([]) # empty array
(np.array([1])==np.array([])).all() # True
np.array([1,2])==np.array([]) # a bool
(np.array([1,2])==np.array([])).all() # error
''' this solution can have a strange behavior in a particular case:
if either A or B is empty and the other one contains a single element,
    then it return True.
    For some reason, the comparison A==B returns an empty array,
    for which the all operator returns True.
Another risk is
if A and B don't have the same shape and aren't broadcastable,
    then this approach will raise an error.'''
np.array_equal(A,B)  # test if same shape, same elements values
np.array_equiv(A,B)  # test if broadcastable shape, same elements values
if A.shape == B.shape:
    print(np.allclose(A,B)) # test if same shape, elements have close enough values
else:
    print(False)
    # official
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
equal = np.allclose(A,B)
print(equal)
    # checking data and shape separately
a = np.random.random((2,3))
b = np.random.random((3,2))
a.data == c.data # this gives total comparison but ignores shape
a.shape == a.shape # this checks the shape only
d = c.reshape(3,2)
a == d
a.data == d.data
a.shape == d.shape and a.data == d.data # this checks both shape and data

# Make an array immutable (read-only)
a = np.arange(10)
print(a.flags)
a.flags.writeable = False
a[0] = 1
# Gives: RuntimeError: array is not writeable
  # wrong
a = np.random.random((2,3))
a
b = tuple(a)
a
a[1,1] = 1234
a
a[1,1] # how this can be possible??????
b[1,1]
b[1][1] # can't change b, but (1) b is not an array (2) can change the array
  # that b refers to

# Consider a random 10x2 matrix representing cartesian coordinates,
# convert them to polar coordinates
    # 1, wrong
a = 1000*np.random.randn(10,2)
a = np.random.randint(-1000,1000,size=10*2).reshape(10,2)
a[:,1] # y coordinates
a[:,0] # x coordinates
a[:,1]/a[:,0] # the tan() values for the angles
tmp = a[:,1]/a[:,0]
np.arctan(a[:,1]/a[:,0]) # the angle values in radians, only I and IV quadrants
b = np.arctan(a[:,1]/a[:,0]) # to make it correct you need to evaluate the quadrant
        # separately so that to set the proper sign
a[:,0]**2
a[:,0]**2 + a[:,1]**2
r = np.sqrt(a[:,0]**2 + a[:,1]**2)  # this is correct
np.vstack((b,r))
np.transpose(np.vstack((b,r)))
    # official, correct
Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)  # Element-wise arc tangent of ``x1/x2`` choosing the quadrant correctly.
print(R)
print(T)
  # np.arctan takes the tan value and returns the angle in [-pi/2, pi/2]
  # np.arctan2 takes y and x in allowing to choose any of the 4 quadrants

# Create random vector of size 10 and replace the maximum value by 0
a = np.random.random(10)
a[a.argmax()] = 0
a

# Create a structured array with x and y coordinates
# covering the [0,1]x[0,1] area
    # structured arrays
    # 1: string argument (default names given)
np.zeros(3, dtype='3int8, float32, (2,3)float64')
np.zeros(10, dtype='2float32') # good size, but default names (no x, y)
np.zeros(10, dtype='2f4') # the same
np.random.random(20).astype(dtype='2float32') # error, applied to single elements
np.random.random((10,2)).astype(dtype='2float32') # error, applied to single elements
    # 2: tuple argument
    # when a structure is mapped to an existing data type.
    # This is done by pairing in a tuple, the existing data type
    # with a matching dtype definition.
x = np.zeros(3, dtype=('i4',[('r','u1'), ('g','u1'), ('b','u1'), ('a','u1')]))
x
x['r'] # 32 bit int split into four 8-bit ints
a = np.zeros(10, dtype=('i4',[('x','i2'), ('y','i2')]))
a
a['x']
a['y']
a[1]
a['x'][1] = 1
a['x']
a
a['y'][1] = 2
a['y']
a
a[1]
a[1]['x'] # doesn't work
a[1]['y'] # doesn't work
a['x'][1] # OK
a['y'][1] # OK
    # 3: List argument
    # the record structure is defined with a list of tuples
np.zeros(3, dtype=[('x','f4'),('y',np.float32),('value','f4',(2,2))])
a = np.zeros(10, dtype=[('x', 'f4'),('y','f4')]) # good!
a
a['x']
a['y']
a['x'][1]
a[1]['x']
a.dtype
a.dtype.names
a.dtype.fields
    # 4: Dictionary argument
    # 4A:
np.zeros(3, dtype={'names':['col1', 'col2'], 'formats':['i4','f4']})
np.zeros(10, dtype={'names':['x','y'], 'formats':['f4','f4']}) # good!
    # 4B:
np.zeros(3, dtype={'col1':('i1',0,'title 1'), 'col2':('f4',1,'title 2')})
a = np.zeros(10, dtype={'x':('f4',0), 'y':('f4',1)}) # what is this offset???
a = np.zeros(10, dtype={'x':('f4',0,'x coord.'), 'y':('f4',1,'y coord')})
a
a.dtype
a.dtype.names
a.dtype.fields
    # accessing multiple fields
x = np.array([(1.5,2.5,(1.0,2.0)),(3.,4.,(4.,5.)),(1.,3.,(2.,6.))],
        dtype=[('x','f4'),('y',np.float32),('value','f4',(2,2))])
x = np.array([(1.5,2.5,(1.0,2.0)),(3.,4.,(4.,5.)),(1.,3.,(2.,6.))])
    # this gives error: it's a sequence, not a flat list
    # creating the right shape
np.array(((0,1),(3,7)))  # good shape and data
np.array([(0,1),(3,7)])  # good shape and data
x = np.array([(0,1),(3,7)], dtype=[('x','i4'),('y','i4')])
x
x['x']
x['y']
x[0]
x[1]
x['x'][0]
x[0]['x']
    # generating data
a = np.random.random((10))*100
x = np.array(a, dtype=[('x','i4'),('y','i4')])
    # the problem: the values are duplicated,so always x == y
a.astype([('x','i4'),('y','i4')])
a.astype({'x':('i4',0), 'y':('i4',1)})
    # strange:
    # y is the same a
    # but x is a shifted by 8 bytes:
    # 4: 1024+4, 4
    # 3: 768+3, 3, 3<<8
    # 20: 5120+20, 20
    # a: a<<8, a
    # Filling structured arrays
arr = np.zeros((5,), dtype=[('var1','f8'),('var2','f8')])
arr
    # 1: by field:
arr['var1'] = np.arange(5)
arr
    # 2: by row:
arr[0] = (10,20) # only a tuple allowed here
arr
arr[1] = ((30,50),(60,80)) # error
    # solution
a = np.zeros(10, dtype=[('x', 'f4'),('y','f4')])
a['x'], a['y'] = np.random.random(10), np.random.random(10)
a
    # attempts
Z = np.zeros((10,10), [('x',float),('y',float)])
Z
Z[1]
Z[1,1]
# official solution
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,10),
                             np.linspace(0,1,10))
np.meshgrid(np.linspace(0,1,10), np.linspace(0,1,10))
  # this one makes
  # carthesian concatenation of n vectors in a matrix of n-size tuples
print(Z)
    # checking
Z = np.zeros((1,1), [('x',float),('y',float)])
Z
Z['x'], Z['y'] = 0.2, 0.3
Z
np.linspace(0,1,2)
np.meshgrid(np.linspace(0,1,2), np.linspace(0,1,2))
# Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,10), np.linspace(0,1,10))
Z

# Given two arrays, X and Y, construct the Cauchy matrix C (Cij = 1/(xi - yj))
x = np.random.random(10)
y = np.random.random(10)
np.fromfunction(lambda i, j: i+j, (10,10))
np.fromfunction(lambda i, j: x[0], (10,10)) # interesting, returns a scalar
np.fromfunction(lambda i, j: x[i], (10,10)) # error
np.fromfunction(lambda i, j: print(i), (10,10))
np.fromfunction(lambda i, j: print(x[i]), (10,10)) # error
np.fromfunction(lambda i, j: 1 / (x[i] - y[j]), (10,10)) # error
np.fromfunction(np.vectorize(lambda i, j: 1 / (x[i] - y[j])), (2,2)) # check
1 / (x[0]-y[0])
1 / (x[0]-y[1])
1 / (x[1]-y[0])
1 / (x[1]-y[1]) # correct
    # solution
np.fromfunction(np.vectorize(lambda i, j: 1 / (x[i] - y[j])), (10,10))
    # official
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y) # how this works?????
C
print(np.linalg.det(C))
np.fromfunction(np.vectorize(lambda i, j: 1 / (X[i] - Y[j])), (8,8))

# Print the minimum and maximum representable value for each numpy scalar type
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)
# checking
np.iinfo(np.int8) # min, max, dtype
np.finfo(np.float32) # resolution, min, max, dtype

# How to print all the values of an array?
# np.set_printoptions(threshold=np.nan)
# Z = np.zeros((25,25))
# print(Z)

# How to find the closest value (to a given scalar) in an array?

a = np.random.randint(100,size=10)
a
b = 50
    # 1
a[np.argmin(abs(a-b))]
    # 2 official
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])
    # check
np.argmin(abs(a-b))
np.abs(a-b).argmin()

# Create a structured array representing a position (x,y) and a color (r,g,b)
np.zeros(1, dtype=[('x','f'),('y','f'),('color','i',3)])
np.zeros(1, dtype=[('x','f'),('y','f'),
                   ('color',[('r','i'),('g','i'),('b','i')])])
np.zeros(1, dtype=[('position',
                    [('x','f'),('y','f')]),
                   ('color',
                    [('r','i'),('g','i'),('b','i')])])
# official
Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                   ('y', float, 1)]),
                    ('color',    [ ('r', float, 1),
                                   ('g', float, 1),
                                   ('b', float, 1)])])
print(Z)
    # the old attempts...
a = np.zeros((10,10,10), dtype = int)
  # wrong: this is a 3-D array of scalars
  # we need a 2-D array of 3-element vectors
b = np.zeros((10,10), [('r', int),('g',int),('b',int)]) # correct
  # let's now populate this with examplar data on points (RGB color for each point)
np.linspace(0,255,10)  # this is float
c = np.linspace(0,255,10).astype(int) # this is int
c
np.meshgrid(c,c,c)
b # 2-D array of 3-element tuples
b['r'], b['g'], b['b']  # 2-D slices, cutting across the tuples
  # solution
Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                   ('y', float, 1)]),
                    ('color',    [ ('r', float, 1),
                                   ('g', float, 1),
                                   ('b', float, 1)])])
print(Z)


# Consider a random vector with shape (100,2) representing coordinates,
# find point by point distances

    # checking
a = 100*np.random.randn(10,2)
a
np.atleast_2d(a) # the same
#
a[:,0] # x coords: one row of data
b = a[:,0]
b # all x coords
b[0] # x coord for point 0
#
np.atleast_2d(a[:,0]) # x coords: one row & 10 columns of data
b = np.atleast_2d(a[:,0])
b[0] # all x coords
b[0,0] # x coord for point 0
a[:,1] # y coords
#
np.atleast_2d(a[:,1])
# np.zeros((2,3))
    # best
x,y = np.atleast_2d(a[:,0],a[:,1])
D = np.sqrt((x-x.T)**2 + (y-y.T)**2)
    # official
Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0], Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)
print(np.sum(D))
    # Much faster with scipy
import scipy # Thanks Gavin Heverly-Coulson (#issue 1)
import scipy.spatial
Z = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z,Z)
print(D)


# How to convert a float (32 bits) array into an integer (32 bits) in place

a = 100 * np.random.random(10)
a
a.astype(int, copy=False)
a # the input array isn't modified, strange...
Z = np.arange(10, dtype=np.int32)
Z = Z.astype(np.float32, copy=False)


# Consider the following file:
#1,2,3,4,5
#6,,,7,8
#,,9,10,11
# How to read it ?

loc = r'D:\data\Dropbox\workarea\python-work\Basic\data'
file = r'np_ex_100_e30.txt'
s = loc+'\\'+file
np.loadtxt(s, dtype=str, delimiter=',') # can't handle missing data
np.genfromtxt(s, delimiter=',', missing_values='') # correct
np.genfromtxt(s, delimiter=',') # correct
    # new file:
## -------------
#1,2,3,4,5
#6,,,7,8
#,,9,10,11
## -------------
file = r'new.txt'
s = loc + '\\' + file
np.genfromtxt(s, delimiter=',') # correct
    # ignorred commmented lines, included nan's

  # checking things
txt = 'test,string,hello'
txt.split(',')
import os
os.getcwd() # just checking where we are
p = os.getcwd() + '\\basic\\data\\np_ex_100_e30.txt'
  # reading the file
f = open(p, 'r')
x = f.read()
f.close()
  # converting into an array
i,j = 0,0
c = np.empty((3,5), dtype = str)  # here a problem: the shape is set in advance
d = np.zeros((3,5), dtype = float)  # here you can use np.nan for missing values
e = np.zeros((3,5), dtype = int)  # missing values have to be set to 0 (not like R where)
for a in x.split('\n'):
  for b in a.split(','):
    # print i, j, b
    c[i,j] = b
    if not b == '':
      d[i,j] = c[i,j].astype(float) # this works for numpy array only
      e[i,j] = int(b)             # this is general python function
    else:
      d[i,j] = np.nan  # missing values
      # e[i,j] # you can't assing None, np.nan etc.
    j+=1
  i+=1
  j=0
print c
print d
print e
  # official solution
p = os.getcwd() + '\\basic\\data\\np_ex_100_e30.txt'
np.genfromtxt(p, delimiter=",")


# What is the equivalent of enumerate for numpy arrays

    # official
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print(index, value)
for index in np.ndindex(Z.shape):
    print(index, Z[index])
    # checking for structured arrays
x = np.array([(1,2.,'Hello'), (2,3.,"World")],
             dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'S10')])
x
for index, value in np.ndenumerate(x):
    print(index, value)


# Generate a generic 2D Gaussian-like array

    # 1
def gaussian2d(x,y,A=1,x0=0,y0=0,sx=1,sy=1):
    return A*np.exp(-((x-x0)**2/(2*sx**2)+(y-y0)**2/(2*sy**2)))
np.fromfunction(gaussian2d, (3,3), x0=1, y0=1)
    # checking
x=np.array([0,1,0,1])
y=np.array([0,0,1,1])
gaussian2d(x,y)
    # official
X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)
    # cmp
X, Y = np.meshgrid(np.linspace(-1,1,3), np.linspace(-1,1,3))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)

  # old: checking
X,Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
X
Y
A = 1 / (sigma * np.sqrt(2*np.pi)) # that should be for Normal density function
  # Gaussian function works for any constant A
G2 = A * np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G2)


''' How to randomly place p elements in a 2D array? '''

    # 1, given up
x,y = 2,3
p = x*y
# x = np.random.randint(0,100,p)
x = np.arange(p)
np.zeros((x,y), dtype=int)
# X, Y = np.meshgrid(np.arange(x),np.arange(y))
    # official
n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
Z
    # checking
np.random.choice(range(n*n), p, replace=False)

  # old: checking value in an array
a = np.arange(8).reshape((-1,2))
[5,8] in a
  # attempt
p = 10
a = np.arange(p)+1
b = np.zeros((p,p), dtype=int)
coord = np.empty((p,2),dtype=int)
coord.fill(-1)
for i in np.arange(p):
  cond = True
  j = 0
  while cond:
    x,y = np.floor(np.random.random(2)*p).astype(int)
    j+=1
    if j > 10:
      break
    print x, y
    if [x,y] in coord:  # this check doesn't work correctly (to many false repetitions)
      print 'repet'
    else:
      print 'ok'
      coord[i,] = x,y
      cond = False
  # official solution
n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
  # checking
a = np.random.choice(range(n*n), p, replace = False)  # this is:
a
  # flat array p*p i.e. 100 element vector of 0,1,...,99
  # from this choose 3 elements, without replacing
  # these will be indeces of the flattened array 10*10 into 100 element vector
np.put(Z, a, 1) # this is:
  # Z is 10*10 array (a map to put the elements onto)
  # a is the list of indeces for flattened Z i.e. 100 elements vector
  # 1 is the element to be put on the array


''' Subtract the mean of each row of a matrix '''

x = np.random.random((3,2))
x
    # 1
y = x - x.mean(0) # deducting mean from each column
y.sum(0)
x - x.mean(1) # error, rows are treated as columns
x - x.mean(1).T # transposition can't help
x - np.atleast_2d(x.mean(1)) # still wrong, but now explicit: one row, 3 cols
x - np.atleast_2d(x.mean(1)).T # now works!
y = x - np.atleast_2d(x.mean(1)).T
y.sum(1)
    # 2
x - x.mean(1)[:,np.newaxis]
    # 3
x - np.expand_dims(x.mean(1),1)
    # 4
x - x.mean(1).reshape(3,1) # not general
x - x.mean(1).reshape(x.mean(1).shape[0], 1) # general but ugly
x - x.mean(1).reshape(-1,1) # unspecified value (more general)
    # 5
z = x.mean(1)
z.shape = (z.shape[0],1)
x - z
    # 6, best!
x - x.mean(axis=1, keepdims=True)

    # old
  # my initial solution
a = np.random.random((4,5))  # 4 rows, 5 cols
a
np.mean(a)  # mean of the whole array
np.apply_along_axis(np.mean, 0, a) # along rows, i.e. mean for each column
m = np.apply_along_axis(np.mean, 1, a) # along cols, i.e. mean for each row
m
a-m  # can't work, broadcasting rules are different
m2 = np.expand_dims(m,1)  # expand dimensions: 1 -> 2
m2
m3 = m2.repeat(5,1) # repeat so that you have the same shape as a
  # and ech cell in row i, you have the average for row i
m3
a_std = a-m3  # done
np.mean(a_std)  # just checking, should be very close to 0, if the mean is deducted
np.apply_along_axis(np.mean, 1, a_std) # just checking, should be very close to 0
  # official solution
X = np.random.rand(5, 10)
# Recent versions of numpy
Y = X - X.mean(axis=1, keepdims=True)
# Older versions of numpy
Y = X - X.mean(axis=1).reshape(-1, 1)
  # checking
a.mean()  # whole array mean
a.mean(0) # mean across rows i.e. for each column
a.mean(1) # mean across cols i.e. for each row
a - a.mean(1) # again a problem, one dimension is lost
a - a.mean(1, keepdims=True)  # now correct, as the orignal number of dimensons is kept
  # and broadcasting rules take care of the deduction
  # equivalent:
a - m2
a - np.expand_dims(np.apply_along_axis(np.mean, 1, a),1)

''' How to I sort an array by the nth column? '''

x = np.random.randint(10, size=(4,5))
np.sort(x) # this sorts elements in each row separately
np.sort(x, 0) # this sorts elements in each column separately
x.sort() # in place, elements in rows
x.sort(0) # in place, elements in cols
x.argsort(0) # indeces of sorted elements in columns
y = x.argsort(0)[:,1] # the indeces for sorting column 1
x[y] # this is actually it, all collumns sorted along with column = col
col = 1
x[x.argsort(0)[:,col]]
    # official
Z = np.random.randint(0,10,(3,3))
print(Z)
print(Z[Z[:,1].argsort()]) # along column 1
Z[:,1]
    # checking
col = 1
x[x.argsort(0)[:,col]]
x[x[:,col].argsort()] # both give the same result

    # old
a = np.random.rand(4,5)*10
a = np.random.random((4,5))*10 # the same
a = a.astype(int)
  # checking
a
np.sort(a)  # default: sorted by axis=-1 i.e. along the last axis
  # i.e. each row contains elements sorted left-right
np.sort(a, None)  # flattened array sorted
np.sort(a, 0)  # sorted by rows
  # i.e. each column contains elements sorted top-down
  # so you need argsort
np.argsort(a)  # now: sorted by axis-1, indeces given
  # so: for each row: sort the elements comparing columns, give the indeces of sorted elements
n = 3
a[:,n]  # n-th column
i = np.argsort(a[:,n])  # indeces for sorting the n-th column
a[i,n]  # sorterd n-th column
  # i.e. a[indeces of rows, index of a col]
a[i]  # the array a sorted by n-th column
  # i.e. a[indeces of rows] (and all columns)
a[i,:]  # the same
  # official
Z = np.random.randint(0,10,(3,3))
Z
Z[Z[:,1].argsort()]
  # checking
a[a[:,n].argsort()] # OK

''' How to tell if a given 2D array has null columns '''

    # checking for columns with NaN's
x = np.random.random((3,5))
x[1,2] = np.nan
x
np.isnan(x) # element wise
np.isnan(x).any() # any Nan? (fastest)
np.isnan(np.sum(x, axis=0)) # this is slow but maybe useful for very big x
np.isnan(x).any(axis=0) # columns that have NaNs
np.isnan(x).any(axis=1) # rows that have NaNs
    # speed test
import timeit
s = 'import numpy;a = numpy.arange(10000.).reshape((100,100));a[10,10]=numpy.nan'
ms = [
    'numpy.isnan(a).any()',
    'any(numpy.isnan(x) for x in a.flatten())']
for m in ms:
    print("  %.2f s" % timeit.Timer(m, s).timeit(1000), m)
    # official
Z = np.random.randint(0,3,(3,10))
print((~Z.any(axis=0)).any())
# checking
Z[:,0]=[0,0,0]
Z
~Z.any(axis=0) # it shows columns with 0's only

    # old
a = np.random.randint(0,100, (3,2))
a
a[0,1] = None  # doesn't work: None, NA
a[0,1] = np.nan # doesn't work, nan is float
b = np.random.random((3,2))
b
b[0,1] = np.nan # works
b
np.sum(b, axis=0) # sum along rows, i.e. sum for each column
  # gives nan if a col contains a nan
  # Getting indeces of not nan elements:
c = np.random.randint(0,2, (3,2))
ind = np.arange(c.size)
c
c==0
np.where(c.flat==0,ind,np.nan)
ind[c.flat==0]
  # final solution:
b2 = np.sum(b, axis=0)
ind = np.arange(b2.size)
ind[np.logical_not(np.isnan(b2))] # you can't use not as not is not vectorized
ind[~(np.isnan(b2))] # OK too
  # official solution
Z = np.random.randint(0,3,(3,10))
print((~Z.any(axis=0)).any())
Z
Z.any(axis=0) # logical vector: if cols contain at least one non-zero value
Z.any(axis=0).any()  # logical scalar: if all calls are not null (see above)

''' Find the nearest value from a given value in an array '''

x = np.random.random((2,2))
a = 0.3
d = np.abs(x-a)
x.flat[d.argmin()] # the value, through a 1-D iterator
x.flatten()[d.argmin()] # the value, through a 1-D copy of the x array
np.unravel_index(d.argmin(),x.shape) # index of the element
x[np.unravel_index(d.argmin(),x.shape)] # the element
    # old
a = np.random.random((4,5))
b = 0.5
a.flat[np.argmin(np.abs(a-b))]
  # official
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])

''' Create an array class that has a name attribute '''

    # failed attempts
class NamedArray(np.ndarray):
    def __init__(self, name):
        self.name =- name
x = NamedArray((1,2),name="It")
    # official solution
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")
Z = NamedArray(np.arange(10), "range_10")
print (Z.name)
Z

''' Consider a given vector, how to add 1 to each element indexed by a second vector
(be careful with repeated indices)? '''

x = np.random.randint(10, size=10)
i = np.random.randint(10, size=10)
    # checking
for j in i:
    print(j)
np.unique(i)
x[1]
x[1] += 1
x[1]
    # solution -> ignore repeated indeces
x
x[np.unique(i)] += 1
x
    # official solution -> include repeated indeces
Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)
    # checking
np.bincount(i) # 0-8, so only 9 bins
np.bincount(i, minlength=len(x)) # 10 bins, as len(x)
x+np.bincount(i, minlength=len(x))

''' How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? '''

x = [1,2,3,4,5]
i = np.random.randint(10, size=5)
F = np.bincount(i, x)
F
    # official
X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print(F)
''' explanation:
You have a list of values (x) that are grouped in k groups. labelled 1...k.
Grouping is done using and index (i) indicating for each position in x,
which group it belongs to.

The task is to get a sum of values for each group.

Solution:
count how many times a group index j appears in the list i,
but don't use '1' for counting but a weight given by the value
at the respective position in vector x.
'''

''' Considering a (w,h,3) image of (dtype=ubyte),
compute the number of unique colors'''

    # official - something is wrong with it - but I don't understand it actually
w,h = 2,2
I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
n = len(np.unique(F))
print(np.unique(I))

''' Considering a four dimensions array, how to get sum over the last two axis at once '''

x = np.random.random((2,3,4,5))
A = np.random.randint(0,10,(3,4,3,4))
# solution by passing a tuple of axes
sum = A.sum(axis=(-2,-1))
print(sum)
# solution by flattening the last two dimensions into one
# (useful for functions that don't accept tuples for axis argument)
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)
    # checking
x.shape
x.shape[:-2] # shape of dimensions 0 and 1 (i.e. all except the last two dims)
x.shape[:-2]+(-1,) # appending -1 at the end of the tuple
x.reshape(x.shape[:-2] + (-1,)) # the -1 means the size of the last dim
# is selected so to accomodate all the data.
# This way the last two dims are flattened into one last dim.
x.reshape(x.shape[:-2] + (-1,)).sum(axis=-1) # sum over the last dim

    # old
# 49. Considering a four dimensions array, how to get sum over the last two axis at once?
a = np.random.randint(0,10,(2,2,2,2))
a[0] # upper 2-D slice
a[1] # lower 2-D slice
a[:,0,...] # slices 0 and 2
a[:,1,...] # slices 2 and 3
a[:,:,0,:] # rows 0 of all slices
a[...,0,:]
a[...,1,:] # rows 1 of all slices
a[...,0]   # cols 0 of all slices, arranged in rows
a[...,1]   # cols 1 of all slices, arranged in rows
np.sum(a) # whole array
np.sum(a, 0) # upper 2-D slice + lower 2-D slice
np.sum(a, 1) # slices 0 and 2 + slices 1 and 3
np.sum(a, (2,3)) # sum across rows and cols, i.e. preserving slices
  # i.e. for each slice we get one value (the sum across all its rows and calls)
  # = an array 2x2, each cell showing sum for each slice of a
  # official - mine is much simpler
A = a.copy()
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)
  # checking
A = a.copy()
A.shape
A.shape[:-2]
A.shape[:-2] + (-1,)
A.reshape(A.shape[:-2] + (-1,)) # stacking up the last two dimensions

''' Considering a one-dimensional vector D, how to compute means of subsets of D
using a vector S of same size describing subset indices? '''

D = np.random.uniform(0,1,100)
S = np.random.randint(10,100)
    # solution using pandas
import pandas as pd
df = pd.DataFrame({'D':D,'S':S})
df.groupby('S').mean()
pd.DataFrame(D).groupby(S).mean()
pd.Series(D).groupby(S).mean() # faster
    # solution using bincount - see the explanation a few exercises above
s = np.bincount(S, D) # getting the sums
c = np.bincount(S) # getting the counts
s/c # getting the means
np.bincount(S, D)/np.bincount(S)
    # speed check
import timeit
timeit.Timer('pd.DataFrame(D).groupby(S).mean()','import pandas as pd')
timeit.timeit() # pandas data frame is slowest
timeit.Timer('np.bincount(S, D)/np.bincount(S)','import numpy as np')
timeit.timeit() # numpy is fastest
timeit.Timer('pd.Series(D).groupby(S).mean()','import pandas as pd')
timeit.timeit() # pandas series is a bit slower than numpy array
    # official
D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)
    # yet another solution (not completed)
np.nanmean(np.where(S==0,D,np.NaN))

    # old
n = 20
s = 5
D = np.random.randint(10, size=n)
i = np.random.randint(s, size=n)
  # checking
i3 = np.where(i == 3)[0]
i[i3]
D[i3]
np.mean(D[i3])
  # done -> in a loop
for j in range(s):
  print j, np.mean(D[np.where(i == j)[0]])
  # mimicking the official solution -> result in one array
D_sums = np.bincount(i, weights=D)
D_counts = np.bincount(i)
print D_sums / D_counts
  # official
D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)

''' How to get the diagonal of a dot product? '''

a = np.random.randint(10, size=(2,3))
b = np.random.randint(10, size=(3,4))
a
b
np.dot(a,b) # not quadratic, has no diagonal actually
# slow
np.diag(np.dot(a,b))
# fast
np.sum(a*b.T, axis=1) # error, as it is not quadratic (has no diagonal actually)
a = np.random.randint(10, size=(3,2))
b = np.random.randint(10, size=(2,3))
a
b
np.dot(a,b)
np.diag(np.dot(a,b))
np.sum(a*b.T, axis=1)
# faster version
np.einsum("ij,ji->i", a, b)

# checking
a = np.arange(25).reshape(5,5)
b = np.arange(5)
c = np.arange(6).reshape(2,3)
a
b
c

np.einsum('ii', a) # sum across the diagnoal
np.einsum(a, [0,0]) # sum of the diagonal
np.einsum(a, [0,1]) # identity
np.einsum(a, [1,0]) # transposition
np.einsum(a, [0,0,0]) # error too many subscripts
np.einsum(a, [1,1]) # sum of the diagonal
np.einsum(a, [0,2]) # identity
np.einsum(a, [2,0]) # transposition
np.einsum(a, [1,2]) # identity
np.einsum(a, [...,0]) # identity
np.einsum(a, [0,...]) # transposition
np.trace(a) # sum of the diagnoal

np.einsum('ii->i', a) # diagnoal
np.einsum(a, [0,0], [0]) # ...
np.diag(a) # ...

np.einsum('ij,j', a, b) # dot product
np.einsum(a, [0,1], b, [1]) # ...
np.dot(a, b) # ...
np.einsum('...j,j', a, b) # ...

np.einsum('ji', c) # transpose
np.einsum(c, [1,0])
c.T
# for more see help

    # old
n = 10
  # wrong approach:
  # a dot product of 1-D array is a scalar
x = np.random.randint(0,10, size=(2,n))
A,B = np.split(x,2) # default axis=0 i.e. vsplit
A,B = np.vsplit(x,2)
np.dot(A,B) # error (too many dimensions)
A.shape
A = A.reshape(n)
A.shape
B.shape
B.resize(n)
B.shape
np.dot(A,B) # now OK
np.diag(np.dot(A, B)) # given a scalar, while an array expected (1-D or 2-D)
  # just checking
Y = np.arange(10)
Y.shape # the same shape
np.diag(Y)
np.diag([238])
A = np.random.randint(0,10, size=n)
B = np.random.randint(0,10, size=n)
np.dot(A,B)
  # correct:
  # using 2-D arrays
A = np.random.randint(0,10, size=(n,n))
B = np.random.randint(0,10, size=(n,n))
np.dot(A, B)
# official
# Slow version
np.diag(np.dot(A, B))
# Fast version
A*B.T
np.sum(A * B.T, axis=1)
# Faster version
np.einsum("ij,ji->i", A, B)

''' Consider the vector [1, 2, 3, 4, 5],
how to build a new vector with 3 consecutive zeros interleaved between each value?'''

    # attempt 1
a = np.arange(1,6)
n = a.shape[0]
m = n*4-3
b = np.zeros(m, dtype=np.int)
c = np.arange(n)
#b = np.arange(m)
b[c*4]
b[c*4] = a
b
    # 1 in short
a = np.arange(1,6)
# b = np.zeros(a.shape[0]*4-3)
# b[np.arange(a.shape[0])*4] = a
b = np.zeros(len(a)*4-3)
b[np.arange(len(a))*4] = a
b
    # np.copyto - too complicated, nonsense...
a = [(1,3),(5,4)]
[i for sub in a for i in sub]
x = np.zeros(10)
x
np.copyto(x, np.arange(10), where=[1,0]*5)
x
list(zip(range(1,6),range(1,6)))
d = [i for sub in list(zip(range(1,6),range(1,6),range(1,6),range(1,6))) for i in sub]
    # 2 np.place
x = np.zeros(10)
x
np.place(x, vals=np.arange(10), mask=[1,0]*5)
x = np.zeros(4*5)
x
np.place(x,[1,0,0,0]*5,np.arange(1,6))
x
    # 2 in short
a = np.arange(1,6)
# b = np.zeros(a.shape[0]*4-3)
b = np.zeros(len(a)*4-3)
np.place(b,([1,0,0,0]*5)[:-3],a)
b
    # 3 np.put
a = np.arange(1,6)
#b = np.zeros(a.shape[0]*4-3)
b = np.zeros(len(a)*4-3)
np.put(b,np.arange(5)*4,a)
b
    # official - essentially the same as my 1, though with nice tweaks
Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)
# checking
np.arange(1,18)[::4+1]
# maybe suggest another solution - based on yours 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
np.put(Z0, np.arange(len(Z))*(nz+1), Z)
print(Z0)

    # old
a = [1, 2, 3, 4, 5]
b = np.zeros(4*4+1, dtype=int)
for i in range(5):
  b[i*4] = a[i]
print b
  # official
Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z  # this is the only different thing: clever indexing instead of looping
print(Z0)
  # checking
c = np.arange(24)
c[::1]
c[::2]
c[::3]
c[::3] = 1
c
c[::3] = [1,2] # broadcasting doesn't apply here
c[::3] = range(2) # the same problem
c[::3] = range(8) # this works, the same shape = (8,) for both objects
c

''' Consider an array of dimension (5,5,3),
how to mulitply it by an array with dimensions (5,5) '''

a = np.random.randint(3, size=(5,5,3))
b = np.random.randint(3, size=(5,5))
b1 = b[:,:,None]
b2 = np.expand_dims(b,2)
b3 = np.atleast_3d(b)
b4 = b[:,:,np.newaxis]
a*b1
a*b2
a*b3
a*b4
np.newaxis is None

    # old
a = np.random.randint(0,10,(5,5,3))
b = np.random.randint(0,10,(5,5))
  # 1
a*b # can't broadcast the shapes
  # 2
np.expand_dims(b, axis=0)  # it's like this option but then need to repeat the (5,5) shaped slices 3 times
np.expand_dims(b, axis=1)
np.expand_dims(b, axis=2)
c1 = np.expand_dims(b, axis=0)
c1.shape  # nope, we need (5,5,3) shaped array
c2 = np.expand_dims(b, axis=2)
c2.shape
d = np.repeat(c2, 3, axis=2) # this is exactly what's needed
a.shape
d.shape
y = a*d
y
y.shape
  # 3
a*c2  # the same, so no need to repeat the content
  # 4, official
A = np.ones((5,5,3))
B = 2*np.ones((5,5))
A*B # can't broadcast the shapes
print(A * B[:,:,None]) # gives (5,5,1) shape, no need to repeat the content
  # checking
b[:]   # just b
b[:,:]  # just b
b[:,:,:] # to many indeces
b[:,:,None] # gets (5,5,1) shape
f = b[:,:,None]
f.shape
a*f

''' How to swap two rows of an array? '''

a = np.random.randint(10, size=(3,5))
a
a[[1,0,2]]
a[:,0]

    # old
a = np.random.randint(0,10,(3,5))
a
# swap 0 and 2
  # 1
b = a.copy()
b[0] = a[2]
b[2] = a[0]
b
  # 2 tuple unpacking - failed, this is done in a sequence
a
a[0], a[2] = a[2], a[0] # this is sequntial, worng result
a  # wrong, we have only a[2] now
  # 3 official
A = np.arange(25).reshape(5,5)
A
A[[0,1]] = A[[1,0]]
print(A)
  # checking
A
A[0,1] # the element at (row 0, col 1)
A[[0,1]] # rows 0,1
A[[0,1],:] # the same
  # 4 reapplying
a = np.random.randint(0,10,(3,5))
a
a[[0,2]] = a[[2,0]]
a  # it works!


''' Consider a set of 10 triplets describing 10 triangles (with shared vertices),
find the set of unique line segments composing all the triangles '''

faces = np.random.randint(100,(10,3)) # 10 triangles, vertices given by numbers
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
G = F.reshape(len(F)*3,2) # a list of line segments, 3 line segments per each triangle
H = np.sort(G,axis=1) # now the order of vertices is sorted, so identical segments have equal representation
I = H.view( dtype=[('p0',F.dtype),('p1',F.dtype)] ) # elemets are now in tuples (called p0 and p1)
# New view of array with the same data
# this creates a new data type
# tuples are necessary for the np.unique() function not to compare vertices but segments
J = np.unique(I) # only uniqe pairs (= line segments) left
print(J)
# checking
len(I)
len(J)
np.unique(G)
[tuple(i) for i in G] # list of tuples
np.unique([tuple(i) for i in G]) # doesn't work, you need to define a type for the elements
    # old
a = np.arange(10)
a
a.repeat(2) # flatten array
a.repeat(2, axis=0) # along the only axis
a.repeat(2, axis=1) # error
np.roll(a,1)
np.roll(a,2)
np.roll(a,-1)

''' Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? '''

C = np.random.randint(0,10,size=10)
    # checking
np.repeat(np.expand_dims(np.arange(10),1),np.arange(10)) # great!
np.repeat(np.expand_dims(np.arange(len(C)),1),C)
np.repeat(np.arange(len(C))[:,None],C)
    # solution (the same as official)
D = np.repeat(np.arange(len(C)),C)
np.bincount(D)
C

    # old
C = np.bincount([1,1,2,3,4,4,6])
  # 1
a = np.arange(C.size)
r = np.empty(0,dtype=int)
for i in a:
  r = np.concatenate((r, np.repeat(i,C[i])))
print r
  # official
A = np.repeat(np.arange(len(C)), C)
print(A)
  # checking officia
len(C)
np.arange(len(C))
np.repeat(np.arange(len(C)), C) # the count is broadcasted! it's for each element
  # checking 1
a
c = r.copy()
c
b = np.repeat(0,C[0])
b
c = np.concatenate((c,b))
c
np.repeat(1,C[1])

''' How to compute averages using a sliding window over an array? '''

a = np.random.randint(10,size=20)
n = 4 # windows size
    # by defining a function with a loop
def avg1(x, n = 2):
    for i in range(n,len(x)+1):
        print(x[i-n:i].mean())
avg1(a)
def avg2(x, n = 2):
    if len(x) >= n:
        res = np.empty(len(x)+1-n)
        for i in range(n,len(x)+1):
            res[i-n] = x[i-n:i].mean()
    return res
avg2(a)
    # official: cumsum
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))
    # checking
ret = np.cumsum(Z, dtype=float)
ret
n = 3
ret[n:]
ret[:-n]
ret[n:] = ret[n:] - ret[:-n]
ret # positions 0,1,...,n-2 are meaning less here, the rest preserves the sliding sums
# n-1 preserves sum of elem 1,...,n-1
ret[n:]
ret[n-1:]
return ret[n - 1:] / n
    # old
a = np.random.randint(10,size=20)
a
def av1(x, n=2): # default window = 2
  m = x.size-n+1
  r = np.empty(m)
  for i in range(m):
    r[i] = np.average(x[i:i+n])
  return r
av1(a)
def av2(x, n=None): # default window = max, always returns an array
  if n is None:
    n = x.size
    m = 1
  else:
    m = x.size-n+1
  r = np.empty(m)
  for i in range(m):
    r[i] = np.average(x[i:i+n])
  return r
av2(a,2)
av2(a)
av2(a,a.size)
def av3(x, n=None): # default window = max (and returns int, not array, if so)
  if n is None:
    r = np.average(x)
  else:
    m = x.size-n+1
    r = np.empty(m)
    for i in range(m):
      r[i] = np.average(x[i:i+n])
  return r
av3(a,2)
av3(a)
av3(a,a.size)
  # official
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))
av1(Z,n=3)
moving_average(a,3)
av1(a,3)
  # checking
n = 3
Z
ret = np.cumsum(Z, dtype=float)
ret
ret[n:]
ret[:-n]
ret[n:] - ret[:-n] # sums of elements: 1,2..,n; 2,3,...,1+n; 3,4,...,2+n
                  # missing the sum of elements: 0,1,...,n-1
ret[n:] = ret[n:] - ret[:-n] # assigning the sums into elements n,n+1,...
ret    # this way the element n-1 contains the missing sum of 0,1,...,n-1
ret[n-1:] # here is the final range, including all sums
ret[n - 1:] / n  # dividing sums by n to get the averages


''' STARTING HERE I WAS MOSTLY INSPECTING THE OFFICIAL SOLUTIONS 
    So these were too difficult, too far away from my skills '''


''' Consider a one-dimensional array Z, build a two-dimensional array
whose first row is (Z[0],Z[1],Z[2])
and each subsequent row is shifted by 1
(last row should be (Z[-3],Z[-2],Z[-1]) '''

Z = np.random.randint(10,size=10)
Z = np.arange(10)
Z
    # attempt 1
Z[None,:]
z2 = np.repeat(Z[None,:],3,axis=0).reshape(3,(len(Z)))
z2 = np.repeat(Z[None,:],3,axis=0)
z2
np.roll(z2,[0,-1,-2],axis=1) # doesn't work
np.roll(z2,-1,1)
    # attempt 1B
Z
z3 = np.repeat(Z[None,:],3,axis=0).flatten()
# size = 10, window = 3
z3[[8,9,10]]
z3[[19,20,21]]
# 30,31,32
s = 10
w = 3
np.arange(s-w+1,s+1)
np.arange(2*s+1-w+1,2*s+1+1)
np.arange(3*s+2-w+1,3*s+2+1)
for i in np.arange(1,w):
    print(np.arange(i*s-w+i,i*s+i))
    # attempt 1B in short, works!
s = len(Z)
w = 3
z3 = np.repeat(Z[None,:],3,axis=0).flatten()
drops = np.array([np.arange(i*s-w+i,i*s+i) for i in np.arange(1,w)])
np.delete(z3,drops).reshape(s-w+1,w)
    # attempt 2, works!
Z[:-2]
np.roll(Z,-1)[:-2]
np.roll(Z,-2)[:-2]
np.vstack([Z[:-2], np.roll(Z,-1)[:-2], np.roll(Z,-2)[:-2]])
np.vstack([Z[:-2], np.roll(Z,-1)[:-2], np.roll(Z,-2)[:-2]]).T
    # attempt 3
np.apply_along_axis(np.roll,0,Z,[0,-1,-2]) # same problem again, roll takes in the whole array
np.apply_along_axis(lambda x: print(x),0,np.array([0,-1,-2]))
    # attempt 4
np.apply_over_axes(lambda a, row: np.roll(a,-row,0), Z, 0)
np.apply_over_axes(lambda a, row: np.roll(a[None,row],-row,1), z2, 0)
np.apply_over_axes(lambda a, row: np.roll(a[None,row],-row,1), z2, 1)
np.apply_over_axes(lambda a, row: np.roll(a[None,row],-row,1), z2, 2)
np.apply_over_axes(lambda a, row: np.roll(a[None,row],-row,1), z2, [0,1])
    # official
from numpy.lib import stride_tricks
def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)
    # old
a = np.random.randint(10, size=10)
a = np.arange(10)
a
n = 3
  # 1
n_rows = a.size-n+1
r = np.zeros((n_rows, 3),dtype=int)
r
for i in np.arange(n_rows):
  #print a[i:i+3]
  r[i] = a[i:i+3]
r
  # official
from numpy.lib import stride_tricks
def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(a, n)
print(Z)

''' How to negate a boolean, or to change the sign of a float inplace '''

    # official
Z = np.random.randint(0,2,100)
Z
np.logical_not(Z, out=Z)
Z

Z = np.random.uniform(-1.0,1.0,100)
np.negative(Z, out=Z)

''' Consider 2 sets of points P0,P1 describing lines (2d) and a point p,
   how to compute distance from p to each line i (P0[i],P1[i]) ? '''

p0 = 10*np.random.random((10,2))
p1 = 10*np.random.random((10,2))
p0 = p0.view(dtype=[('x',p0.dtype),('y',p0.dtype)])
p1 = p1.view(dtype=[('x',p1.dtype),('y',p1.dtype)])
p = np.random.random((1,2))
p = p.view(dtype=[('x',p.dtype),('y',p.dtype)])
p0
p1
p
def distance(x1,x2,y1,y2,x0,y0):
    num = np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    den = np.sqrt((y2-y1)**2 + (x2-x1)**2)
    return num/den
distance(0,2,0,2,2,1) # correct
for i in np.arange(len(p0)):
    print(distance(p0[i]['x'],p1[i]['x'],p0[i]['y'],p0[i]['y'],p['x'],p['y']))

    # official
def distance(P0, P1, p):
    T = P1 - P0 # lengths in x and y axis
    L = (T**2).sum(axis=1) # by pythagorean: array of squared lengths
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L # ok, it's getting less intuitive
    U = U.reshape(len(U),1) # 1 row, many cols -> many rows, 1 col
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))
P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))

''' Consider an arbitrary array, write a function that extract a subpart
 with a fixed shape and centered on a given element (pad with a fill value when necessary) '''

a = np.random.randint(10, size=(2,3))
def extract(x):
    pass

    # official
Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)

''' Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? '''

Z = np.arange(1,15)
from numpy.lib import stride_tricks
    # solution using another exercise solution
def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
rolling(Z,4)
    # extract the code line
stride_tricks.as_strided(Z, (len(Z)-3,4))
stride_tricks.as_strided(Z, (len(Z)-3,4),(4,4))
# honestly I don't know how this function works

''' Compute a matrix rank '''

Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
# rank:
np.sum(S > 1e-10)
np.linalg.matrix_rank(Z)

''' How to find the most frequent value in an array? '''

Z = np.random.randint(10, size=10)
Z
np.bincount(Z).argmax()

''' Extract all the contiguous 3x3 blocks from a random 10x10 matrix '''

from numpy.lib import stride_tricks
Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)

''' Create a 2D array subclass such that Z[i,j] == Z[j,i] '''

# Author: Eric O. Lebigot
# Note: only works for 2d array and value setting using indices
class Symetric(np.ndarray): # corrected
    def __setitem__(self, index, value):
        i, j = index
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)
def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)
S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print(S)

# tuples unpacking
def test1(*args):
    print(args)
    for i in args:
        print(i)
def test2(args):
    print(args)
    for i in args:
        print(i)
i,j = (1,2)

''' Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1).
How to compute the sum of the p matrix products at once? (result has shape (n,1))? '''

p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0,2],[0,1]])
S
# It works, because:
# M is (p,n,n)
# V is (p,n,1)
# Thus, summing over the paired axes 0 and 0 (of M and V independently),
# and 2 and 1, to remain with a (n,1) vector.
# checking
'a'*2
'a'+'b'
a = np.array(range(1, 9)).reshape((2,2,2))
A = np.array(('a', 'b', 'c', 'd'), dtype=object).reshape((2,2))
a
A
np.tensordot(a,A)
np.tensordot(a, A, 1)
np.tensordot(a, A, 0)
np.tensordot(a, A, (0, 1))
np.tensordot(a, A, (2, 1))
np.tensordot(a, A, ((0, 1), (0, 1)))
np.tensordot(a, A, ((2, 1), (1, 0)))

''' Consider a 16x16 array, how to get the block-sum (block size is 4x4)? '''

Z = np.ones((16,16), dtype=np.int)
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(S)
# checking 1
x = np.arange(20)
x
# check 0
np.add.reduceat(x,[0])
np.add.reduce(x,0)
x.sum()
# check 2,4
np.add.reduceat(x,[2,4]) # default: axis = 0
x[2:4].sum()
x[4:x.shape[0]].sum()
# check blocks size 4
np.arange(0,len(x),4) # see steps: 0, 4, 8, 12, 16
np.add.reduceat(x, np.arange(0,len(x),4))
# checking 2
Z
Z1 = np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0) # reducing rows (axis 0)
Z1
Z2 = np.add.reduceat(Z1, np.arange(0, Z.shape[1], k), axis=1) # reducing cols (axis 1)
Z2

''' How to implement the Game of Life using numpy arrays? '''

# Author: Nicolas Rougier
def iterate(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])
    print('N\n', N)
    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z
Z = np.random.randint(0,2,(50,50))
Z
for i in range(100): Z = iterate(Z)
print(Z)
# checking
Z = np.random.randint(0,2,(4,4))
Z
for i in range(1):
    Z = iterate(Z)
    print('Z\n',Z)
print(Z)
# -> looks like it doesn't work properly
# corrected version
def iterate(Z):
    # Count neighbours
    X = np.zeros([x+2 for x in Z.shape], dtype=Z.dtype)
    X[1:-1,1:-1] = Z
    N = (X[0:-2,0:-2] + X[0:-2,1:-1] + X[0:-2,2:] +
         X[1:-1,0:-2]                + X[1:-1,2:] +
         X[2:  ,0:-2] + X[2:  ,1:-1] + X[2:  ,2:])
    # Apply rules
    birth = (N==3) & (Z==0)
    survive = ((N==2) | (N==3)) & (Z==1)
    Z[...] = 0
    Z[birth | survive] = 1
    return Z
Z = np.random.randint(0,2,(50,50))
Z
for i in range(100): Z = iterate(Z)
print(Z)

''' How to get the n smallest value of an array '''

a = np.arange(10)
a
np.random.shuffle(a) # permutation in place
a
#np.random.permutation(a) # returns a copy
n = 3
n = n-1 # the nth smallest (e.g. 3rd) is indexed n-1 (e.g. 2), as indexing starts at 0
# 1
a.sort() # sorts in place
a[n]
# 2
np.sort(a)[n] # returns a sorted copy
# 4
a[a.argsort()[n]] # returns indeces sorting the array (order() in R)
# 5
a.partition(n) # partition in place
a[n]
# 6
np.partition(a,n)[n] # returns a partitioned copy of the array (kth elem. is at the right position, smaller/bigger values to the left/right of it)
# 7
a[a.argpartition(n)][n]

''' How to get the n largest values of an array '''

a = np.arange(10)
a
np.random.shuffle(a) # permutation in place
a
#np.random.permutation(a) # returns a copy
n = 3
n = n # here last element is indexed -1, second last -2, ... so last 3 are (-3,-2,-1)
# 1
a.sort() # sorts in place
a[-n:]
# 2
np.sort(a)[-n:] # returns a sorted copy
# 4
a[a.argsort()[-n:]] # returns indeces sorting the array (order() in R)
# 5
a.partition(-n) # partition in place
a[-n:]
# 6
np.partition(a,-n)[-n:] # returns a partitioned copy of the array (kth elem. is at the right position, smaller/bigger values to the left/right of it)
# 7
a[a.argpartition(-n)][-n:]
# official
a[np.argpartition(-a,n)[:n]]
# -> here we actually collect (n+1) largest elements
#   it's enought to collect (n) largerst elements:
a[np.argpartition(-a,n-1)[:n]]

np.partition(a,-n)

''' Given an arbitrary number of vectors, build the cartesian product (concatenation, not multiplication)
    (every combinations of every item) '''

def cartesian(arrays): # accepts 1-D arrays (vectors)
    arrays = [np.asarray(a) for a in arrays] # a tuple of array-like objects into a list of numpy arrays
    shape = (len(x) for x in arrays) # an iterator of array lengths
    ix = np.indices(shape, dtype=int) # indices covering all elements in a hierarchical order:
        # an array of len(x) of A slices (len(x[0])) of B rows (len(x[1])) of C columns (len(x[2]))
    ix = ix.reshape(len(arrays), -1).T # len(x) rows, all else flatten; transposed
    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]
    return ix
print (cartesian(([1, 2, 3], [4, 5], [6, 7])))
# checking
arrays = ([1, 2, 3], [4, 5], [6, 7])
arrays = [np.asarray(a) for a in arrays]
arrays
shape = (len(x) for x in arrays)
list(shape) # the generator needs to be restarted after every usage
shape = (len(x) for x in arrays)
ix = np.indices(shape, dtype=int)
ix
ix2 = ix.reshape(len(arrays),-1)
ix2
# row i: indices of elements of array x[i]
# columns: cover all combinations of all elements from the arrays (one element from every array, so sets of 3 here)
ix3 = ix2.T
ix3
for n, arr in enumerate(arrays):
    print(n, arr)
ix3[:,0]
ix3[:,1]
ix3[:,2]
for n, arr in enumerate(arrays):
    print(n, arrays[n][ix3[:, n]])
ix4 = np.zeros(ix3.shape, dtype=int)
for n, arr in enumerate(arrays):
    ix4[:, n] = arrays[n][ix3[:, n]]
ix4
# checking meshgrid
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
h = plt.contourf(x,y,z)
# compare:
x.shape
y.shape
xx.shape
yy.shape
# the same without meshgrid
z2 = np.sin(x[None,:]**2 + y[:,None]**2) / (x[None,:]**2 + y[:,None]**2)
h = plt.contourf(x,y,z2)
# checking
np.indices([2,2,2,2])

''' How to create a record array from a regular array? '''

Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T, # create a record array from a (flat) list of arrays
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)

''' Consider a large vector Z, compute Z to the power of 3 using 3 different methods '''

x = np.random.rand(5e7)
x.nbytes # size in bytes
x.nbytes/1024**2 # size in MB

x = np.array([2,3,4])
%timeit x**3
%timeit x*x*x
%timeit np.power(x,3)
%timeit np.einsum('i,i,i->i',x,x,x)

''' Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A
that contain elements of each row of B regardless of the order of the elements in B? '''

A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))
B = np.arange(4).reshape(2,2)
B = np.array([[2,3],[0,0]])

# loop solution
r = np.empty(0,dtype=np.int)
for i in np.arange(A.shape[0]):
    ok = True
    for j in np.arange(B.shape[0]):
        for k in np.arange(B.shape[1]):
            if B[j,k] in A[i,:]:
                print('B[',j,',',k,'] =',B[j,k],'found in A[',i,',:]=',A[i,:])
                break
        else:
            print('No elements of B[',j,',:] =',B[j,:],'found in A[',i,',:]=',A[i,:])
            break
    else:
        r = np.append(r,[i])
        print('Every row of B has an element in A[',i,',:]=',A[i,:])
print(r)
# inspired by the first line of the official solution
A[..., np.newaxis, np.newaxis] # shape 8,3 -> 8,3,1,1,
A[..., np.newaxis, np.newaxis] == B # every elemnt of A is compared with every element of B
# so that for every element of A there's a subarray of shape (2,2) = shape of B
# with True/False corresponding to B elements that are equal to the selected A element.
# Conclusion:
# To meet the condition in the question
# each submatrix for a complete row of B (so a set of 3 submatrices)
# You should do
# (1) OR accross rows of the submatrices, and
# (2) OR across slices (3rd dim),
# (3) then AND across columns -> True would mean the condition in question is met.
C = (A[..., np.newaxis, np.newaxis] == B)
C[0,...] # table for row 0 of A
C[0,...].any(axis=2) # here: rows (=elements of row 0 of A), cols (=rows of B),
# value (=the element of row 0 of A equals an element from a row B)
C[0,...].any(axis=2).any(axis=0) # Does any of the elements of row 0 A equals to any element of a row B (row B = column here)
C[0,...].any(axis=2).any(axis=0).all() # the answer for row 0 of A
# now get a array showing True/False for all rows of A: see the steps redone:
C.any(axis=3)
C.any(axis=3).any(axis=1)
C.any(axis=3).any(axis=1).all(axis=1) # now only get the conditions
np.where([True,False,True])
np.nonzero([True,False,True])
# solution
np.nonzero(C.any(axis=3).any(axis=1).all(axis=1))
# melting down
np.nonzero(C.any(axis=(3,1)).all(axis=1))
np.where(C.any((3,1)).all(1))
np.where(C.any((3,1)).all(1))[0]
(C.any((3,1)).all(1)).nonzero()[0]
# official
C = (A[..., np.newaxis, np.newaxis] == B)
rows = (C.sum(axis=(1,2,3)) >= B.shape[1]).nonzero()[0]
print(rows) # different answer!
# checking
C.sum(axis=(1,2,3)) # for every row of A, how many equlities there are between every element of that row and every element of B
B.shape[1] # number of columns
C.sum(axis=(1,2,3)) >= B.shape[1] # rows of A, that have more equalities than B has columns (???)
(C.sum(axis=(1,2,3)) >= B.shape[1]).nonzero() # indeeces of these rows
# This answers to a following problem:
# Find rows of A that contain any elements of B in a number of 2 or more;
# additionally number that occure n times in B, count n times here.
# e.g. B = [0,1][2,2]
# then any row of A containig (at least) 2x 0 or 2x1 or 1x2
# while it should contain (0,2),(1,2)

# create this example and submit an issue
B = np.array([[0,1],[2,2]])
A = np.array([[3,3,3],[0,1,3],[0,0,3],[1,1,3],[2,3,3],[0,2,3],[1,2,3]])
C = (A[..., np.newaxis, np.newaxis] == B)
# criticised solution
rows = (C.sum(axis=(1,2,3)) >= B.shape[1]).nonzero()[0]
print(rows)
# suggested solution
rows = np.where(C.any((3,1)).all(1))[0]
print(rows)

''' Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) '''

a = np.random.randint(0,2,(10,3))
a
# ideas
# unique() on rows have size 1
# np.all(row == first element of the row), or allclose()
# (row == first element of the row).all()
# .max() == .min()
# .max() - .min is 0 (or very small)
a.max()
a.max(axis=0) # for every column
a.max(axis=1) # for every row
a.max(axis=1) != a.min(axis=1)
a[a.max(axis=1) != a.min(axis=1),:] # solution
# np.unique(axis=1) # doesn't work with axes
# official solution
Z = np.random.randint(0,2,(10,3))
Z
Z=a
E = np.logical_and.reduce(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(Z)
print(U)
# checking
Z[:,1:] # cols 1-2
Z[:,:-1] # cols 0-1
Z[:,1:] == Z[:,:-1] # if col 0 == col 1, col 1 == col 2
# Reduces `a`'s dimension by one, by applying ufunc along one axis.
np.logical_and.reduce(Z[:,1:] == Z[:,:-1], axis=1) # rows such that col 0 == col 1 and col 2 == col 2
np.all(Z[:,1:] == Z[:,:-1], axis=1) # the same, much easier

# I suggest the following solution:
print(Z)

# solution for arrays of all dtypes (including string arrays and record arrays)
E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(U)

# soluiton for numerical arrays only, will work for any number of columns in Z
U = Z[Z.max(axis=1) != Z.min(axis=1),:]
print(U)

# solution for any number of columns & any type of elements
a[~(a[:,0,None]==a[:,1:]).all(axis=1)]

# checking solutions for arrays of chars or record arrays

# string array
# official
Z = np.array(list('Hello world!'))
Z = np.array([['a','bb','ccc'],['dd','dd','dd']])
E = np.logical_and.reduce(Z[:,1:] == Z[:,:-1], axis=1) # works OK!
E = np.all(Z[:,1:] == Z[:,:-1], axis=1) # works OK! -> so you can replace with all()
U = Z[~E]
print(Z)
print(U)
# my solution
Z[~(Z[:,0,None]==Z[:,1:]).all(axis=1)] # works OK

# record arrays
# offical
Z = np.array([(1,2.,'Hello'), (2,3.,"World")],
              dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'S10')])
Z = np.array([[(1,2.,'Hello'), (2,3.,"World"), (3,4.,"!")],
               [(1,2.,'ddd'), (1,2.,"ddd"), (1,2.,"ddd")]], # this row contains equal elements
              dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'S10')])
E = np.logical_and.reduce(Z[:,1:] == Z[:,:-1], axis=1) # works OK!
E = np.all(Z[:,1:] == Z[:,:-1], axis=1) # works OK, use this one
U = Z[~E]
print(Z)
print(U)
# my solution
Z[~(Z[:,0,None]==Z[:,1:]).all(axis=1)] # works OK




'''  Convert a vector of ints into a matrix binary representation '''

x = np.array([2**i for i in range(10)],dtype=np.int)
x
x = I
# loop solution
# use log base change: log.2 of x is ln(x)/ln(2)
n = x.size
np.log(x)/np.log(2)
m = int(np.max(np.log(x)/np.log(2))) + 1
m
y = np.zeros((n,m),dtype=np.int)
y
for i in np.arange(n):
    for j in np.arange(m):
        #print(i, j, 'x=',x[i],'bit=',2**j)
        # version 1
        y[i,j] = bool(np.bitwise_and(x[i],2**j)) # notation: left to right (revresed)
        # version 2
        # y[i,j] = np.bitwise_and(x[i],1)
        # x[i] = np.right_shift(x[i],1)
print(y)
# numpy version
# offical solution
I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
B # notation: left to right (revresed)
B = ((I[:, None] & (2**np.arange(8))) != 0).astype(int) # the same
B # notation: left to right (revresed)
print(B[:,::-1]) # correct
# checking
I # a vector (a single row)
I.reshape(-1,1) # single column array
2**np.arange(8) # single row powers of 2
for i in range(I.size): print(format(I[i],'b'))
for i in range(8): print(format(2**i,'b'))
I.reshape(-1,1) & (2**np.arange(8)) # bitwise and - selecting bits for every number given
(I.reshape(-1,1) & (2**np.arange(8))) != 0
# official 2
I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1)) # only unsigned int accepted
print(np.unpackbits(I[:, np.newaxis], axis=1)[:,::-1]) # notation: left to right (revresed)
# checking
?np.unpackbits
I[:, np.newaxis] # a single column
np.unpackbits(I[:, np.newaxis], axis=1) # bits unpacked
np.unpackbits(I, axis=0).reshape(-1,8) # flatten array needs reshaping # the same
np.unpackbits(I).reshape(-1,8) # ditto

''' Given a two dimensional array, how to extract unique rows? '''

a = np.random.randint(0,2,(5,3))
a = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[0,0,1],[2,3,4],[1+2**8,1+2**8+2**16,1+2**8+2**16+2**30],[2**30-1,0,0]])
a
# official
b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
_, idx = np.unique(b, return_index=True)
a[idx]
# checking
?np.ascontiguousarray(a) # returned in C-order, contiguous (row-major order)
?np.void # base class for numpy scalar types
a.dtype.itemsize # 4 bytes per array element
a.shape[1] # 3 columns
(np.void, a.dtype.itemsize * a.shape[1]) # dtype = np.void, sized 12 bytes (so covering one whole row)
np.ascontiguousarray(a).view((np.void, a.dtype.itemsize * a.shape[1]))
b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
b # this is one column - many rows array (vertical vector)
?np.unique
np.unique(b) # flatten b, unique elements of b only
np.unique(b, return_index=True) # returns also indeces giving unique elements
_, idx = np.unique(b, return_index=True)
idx # index of unique elements of b = unique rows of a
a[idx] # selects only the unique rows
a[idx,:] # the same
# alternative
b = np.ascontiguousarray(a).view((np.void, a.dtype.itemsize * a.shape[1]))
np.unique(b).view(a.dtype).reshape(-1,a.shape[1])

''' Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function '''

# two 1-D vectors
A = np.random.randint(10, size=3)
B = np.random.randint(10, size=3)
A
B
np.inner(A,B) # dot product
np.dot(A,B)
np.einsum('i,i->',A,B)
np.einsum('i,i',A,B)
np.outer(A,B) # Cartesian (outer)
np.einsum('i,j->ij',A,B) # correct!
np.einsum('i,j->',A,B) # wrong: this is np.sum(np.outer(A,B))
np.einsum('i,j',A,B) # correct!
# rules:
# 'i' == labelling by i (no summation)
# 'i,i' (repeated) == summation (i collapses)
# 'i ->' == forcing summation
# 'i,i->i' == forcing labelling (no summation)
# e.g. elementwise multiplication
A*B
np.multiply(A,B)
np.einsum('i,i->i',A,B) # multiplied but not summed
A+B # elementwise
np.einsum('i,j->i',A,B) # don't know how to do that...
np.sum(A) # just summing up a vector
np.einsum('i->',A)
# one 2-D array
C = np.outer(A,B)
C
np.einsum('ij',C) # itself
np.einsum('ij->',C) # summed all over
np.einsum('ij->ij',C) # itself
np.einsum('ij->ji',C) # transposed
np.einsum('ii',C) # sum over the diagonal
np.einsum('ii->',C) # sum over the diagonal
np.einsum('ii->i',C) # the diagonal
np.einsum('ij->i',C) # sum preserving the rows (i), collapsing columns (j)
np.einsum('ij->j',C) # sum preserving the columns (j), collapsing rows (i)
# three 1-D vectors
C = np.random.randint(10, size=3)
A
B
C
np.einsum('i,i,i',A,B,C)
np.einsum('i,j,k',A,B,C)
np.einsum('i,j,k->ik',A,B,C)
#
C = np.ones(3, dtype=int)
np.einsum('i,j',A,C)
np.einsum('i,j->',A,C)
np.einsum('i,j->i',A,C)
np.einsum('i,j->j',A,C)
np.einsum('i,i',A,C)
np.einsum('i,i->',A,C)
np.einsum('i,i->i',A,C)
#
np.einsum('i,i,i',A,C,B)
np.einsum('i,i,j',A,C,B)
# nothing here...
np.einsum('i,i,j,j->ij',A,[1,1,1],[1,1,1],B)
np.einsum('i,j,j,i->ij',A,[1,1,1],[1,1,1],B)
np.einsum('i,j,k,l->ijkl',A,[1,0],[0,1],B)

''' Considering a path described by two vectors (X,Y),
    how to sample it using equidistant samples '''

# setting up data
phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)
# checking the data
import matplotlib.pyplot as plt
plt.plot(phi,x)
plt.plot(phi,y)
plt.plot(phi,x,phi,y) # two lines
plt.plot(phi,x,'r--',phi,y,'ys') # red dashed line, yellow squares
plt.plot(x,y)
# equidistant sampling
dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
plt.plot(phi[:-1],dr) # density of points decreases away from the origin
r = np.zeros_like(x) # array of zeros
r[1:] = np.cumsum(dr) # integrate path
plt.plot(phi,r) # each point is further away + the step increases (= density decreases)
# -> this is not a distance from the origin.
#   This is a distance from the first point measured along the spiral created by the points
#   (distance of point i = sum of length of all consecuitive segments from point 0 to point i).
r_int = np.linspace(0, r.max(), 200) # regular spaced path
plt.plot(r_int)
plt.plot(r, x)
# Interpolation of x on r_int
x_int = np.interp(r_int, r, x) # integrate path
# -> This is interpolation.
#   315 data points of (r, x) in (0-500, -30-+40)
#   are intrapolated into 200 data points of (r_int, x_int) in the same range
r.shape, x.shape # 315 data points
r_int.shape, x_int.shape # 200 data points
plt.plot(r_int,x_int)
# interpolation of y on r_int
y_int = np.interp(r_int, r, y)
# -> the same for y.
plt.plot(r_int, y_int)
# I am not sure what 'integrate' means here.
# Sampling means interpolating = extrapolating from one sample to another sample, assuming smooth population
# I don't know why r (path integration) was needed here.

''' Given an integer n and a 2D array X, select from X the rows
which can be interpreted as draws from a multinomial distribution with n degrees,
i.e., the rows which only contain integers and which sum to n.  '''

n = 10
a = np.random.randint(0,5, size=(10,5))
n
a
# my solution
(a>=0).all(axis=1) # rows with only nonnegative elements
a.sum(axis=1)==n # rows where the sum of elements is equal n
np.logical_and(a.sum(axis=1)==n, (a>=0).all(axis=1)) # can't use 'and' (scalars only) or '&' (bitwise ops)
a[a.sum(axis=1)==n]
a[np.logical_and(a.sum(axis=1)==n, (a>=0).all(axis=1))]
# official solution
X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1) # rows that has integers only
~np.mod(X,1).any(axis=1) # the same by me
M &= (X.sum(axis=-1) == n) # ...and rows that sum up to n
print(X[M])
# combined solution
M = ~np.mod(X,1).any(axis=1) # only integers (round numbers)
M &= (a>=0).all(axis=1)) # only nonnegative
M &= X.sum(axis=1) == n # rows of sum equal n
X[M]

''' Compute bootstrapped 95% confidence intervals for the mean of a 1D array X
(i.e., resample the elements of an array with replacement N times,
compute the mean of each sample, and then compute percentiles over the means). '''

N = 20
X = (100*np.random.randn(N)).astype(int)
X
B = 100
# loop solution
from scipy.misc import comb
from matplotlib import pyplot as plt
mn = np.empty(B)
for i in np.arange(B):
    xi = np.random.choice(X, size=N)
    mn[i] = np.mean(xi)
plt.hist(mn)
sns.kdeplot(mn, bw=5)
np.percentile(mn,[2.5,97.5])
# compare with X
sns.kdeplot(X, bw=20)
# official solution
X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)





























''' MATH '''



from sympy import solve, symbols,roots, solve_poly_system
from sympy import *
x = symbols('x')
y = symbols('y')
#x,y = symbols('x','y')
solve(x**3 + 2*x + 3, x)
#a, b, c,d = symbols('abcd')
a = symbols('a')
b = symbols('b')
c = symbols('c')
d = symbols('d')
solve_poly_system([a+b+c+d-1,8*a+4*b+2*c+d-5,27*a+9*b+3*c+d-14,64*a+16*b+4*c+d-30],a,b,c,d)

# einsum
# http://stackoverflow.com/questions/26089893/understanding-numpys-einsum
# It does not build a temporary array of products first; it just sums the products as it goes.
# This can lead to big savings in memory use. In short:
# - repeating a label on LHS -> multiply axeswise (elementwise)
# - differentiated labels on LHS -> cartesian multiplication (axis across axis)
# - ommiting a label on RHS -> sum along this axis
A = np.array([0, 1, 2])
B = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])
# multiply A and B element-wise and then sum along the rows of the new array
A # a single row (of 3 elem.)
A[:, np.newaxis] # a single column, 3 rows
A[:, np.newaxis] * B # A is broadcast from (3,1) to (3,4)
(A[:, np.newaxis] * B).sum(axis=1) # sum over columns (reduce/collapse columns) -> 1 row of 3 elems.
np.einsum('i,ij->i', A, B)
# Rules:
#Here is what happens next:
#
# A has one axis; we've labelled it i. And B has two axes; we've labelled axis 0 as i and axis 1 as j.
# By repeating the label i in both input arrays, we are telling einsum that these two axes should be multiplied together. In other words, we're multiplying array A with each column of array B, just like A[:, np.newaxis] * B does.
# Notice that j does not appear as a label in our desired output; we've just used i (we want to end up with a 1D array). By omitting the label, we're telling einsum to sum along this axis. In other words, we're summing the rows of the products, just like .sum(axis=1) does.
# That's basically all you need to know to use einsum. It helps to play about a little; if we leave both labels in the output, 'i,ij->ij', we get back a 2D array of products (same as A[:, np.newaxis] * B). If we say no output labels, 'i,ij->, we get back a single number (same as doing (A[:, np.newaxis] * B).sum()).
#












