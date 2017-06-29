# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:36:58 2017
Computational Statistics in Python
http://people.duke.edu/~ccc14/sta-663/index.html
(for exercises - see a separate file)

Content
Optimization
"""
import numpy as np
import matplotlib.pyplot as plt
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
import matplotlib.pyplot as plt
from scipy import optimize as opt
from scipy.spatial.distance import pdist, squareform
n = 20
k = 1 # spring stiffness
P0 = np.random.uniform(0, 5, (n,2))
# plot the points
plt.scatter(P0[:,0],P0[:,1])
A = np.ones((n, n))
# n x n array, filled with 1
A[np.tril_indices_from(A)] = 0 # connectivity matrix
# -> Return the indices for the lower-triangle of arr -> set them to 0
L = A.copy()
L # these are the resting lengths
# -> all springs have resting length=1, so the Length matrix == Connection matrix
def energy(P):
    P = P.reshape((-1, 2))
    D = squareform(pdist(P))
    # Converts a vector-form distance vector to a square-form distance matrix, and vice-versa.
    return 0.5*(k * A * (D - L)**2).sum()
# energy(P0.ravel())
energy(P0) # the same
# fix the position of the first few nodes just to show constraints
fixed = 4
# plot the fixed points
plt.scatter(P0[fixed:,0], P0[fixed:,1], c='b'); plt.scatter(P0[:fixed,0], P0[:fixed,1], c='r')
bounds = (np.repeat(P0[:fixed,:].ravel(), 2).reshape((-1,2)).tolist() +
          [[None, None]] * (2*(n-fixed)))
# bounds[:fixed*2+4]
bounds[:fixed*2] # no need to add 4 as the upper bound = last index + 1
sol = opt.minimize(energy, P0.ravel(), bounds=bounds)
plt.scatter(P0[:, 0], P0[:, 1], s=25)
P = sol.x.reshape((-1,2))
plt.scatter(P[:, 0], P[:, 1], edgecolors='red', facecolors='none', s=30, linewidth=2);

"""
Optimization of standard statistical models

http://people.duke.edu/~ccc14/sta-663/BlackBoxOptimization.html#optimization-of-standard-statistical-models

When we solve standard statistical problems, an optimization procedure similar to the ones discussed here is performed. For example, consider multivariate logistic regression - typically, a Newton-like alogirhtm known as iteratively reweighted least squares (IRLS) is used to find the maximum likelihood estimate for the generalized linear model family. However, using one of the multivariate scalar minimization methods shown above will also work, for example, the BFGS minimization algorithm.

The take home message is that there is nothing magic going on when Python or R fits a statistical model using a formula - all that is happening is that the objective function is set to be the negative of the log likelihood, and the minimum found using some first or second order optimzation algorithm.
"""
import statsmodels.api as sm
import pandas as pd
"""
Logistic regression as optimization
"""
df_ = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv") # error
df_ = pd.read_csv('https://stats.idre.ucla.edu/stat/data/binary.csv')
#import os; os.getcwd()
#df_.to_csv('Study/data/binary.csv', index=False)
df_.head()
# We will ignore the rank categorical value
cols_to_keep = ['admit', 'gre', 'gpa']
df = df_[cols_to_keep]
df.insert(1, 'dummy', 1)
df.head()
# -> obtain the same using patsy/dmatrices - see below
"""
Solving as a GLM with IRLS
This is very similar to what you would do in R, only using Python’s statsmodels package. The GLM solver uses a special variant of Newton’s method known as iteratively reweighted least squares (IRLS), which will be further desribed in the lecture on multivarite and constrained optimizaiton.
"""
# formula
fo = 'admit ~ gre + gpa'
fo = 'admit ~ ' + ' + '.join(df.columns[2:])
fo = 'admit ~ %s' % ' + '.join(df.columns[2:])
model = sm.GLM.from_formula(fo, data=df, family=sm.families.Binomial())
# -> replace this with an up-to-date API call - see below
fit = model.fit()
fit.summary()

"""
Solving as logistic model with bfgs
Note that you can choose any of the scipy.optimize algotihms to fit the maximum likelihood model. This knows about higher order derivatives, so will be more accurate than homebrew version.
"""
model2 = sm.Logit.from_formula(fo, data=df)
fit2 = model2.fit(method='bfgs', maxiter=100)
fit2.summary()

"""
up-to-date statsmodels
"""
# using patsy / dmatrices
from patsy import dmatrices
import statsmodels.api as sm
y, X = dmatrices(fo, df_) # return_type='dataframe' if needed
sm.GLM(y, X, sm.families.Binomial()).fit().summary()

# using formula call
import statsmodels.formula.api as smf
smf.glm(fo, df_, family=sm.families.Binomial()).fit().summary()

# using formula call and bfgs optimization
smf.logit(fo, df_).fit(method='bfgs', maxiter=100).summary()

"""
Home-brew logistic regression using a generic minimization function
This is to show that there is no magic going on - you can write the function to minimize directly from the log-likelihood eqaution and run a minimizer. It will be more accurate if you also provide the derivative (+/- the Hessian for seocnd order methods), but using just the function and numerical approximations to the derivative will also work. As usual, this is for illustration so you understand what is going on - when there is a library function available, youu should probably use that instead.
"""
def f(beta, y, x):
    """Minus log likelihood function for logistic regression."""
    return -((-np.log(1 + np.exp(np.dot(x, beta)))).sum() + (y*(np.dot(x, beta))).sum())
beta0 = np.zeros(3)
opt.minimize(f, beta0, args=(df['admit'], df.ix[:, 'dummy':]), method='BFGS', options={'gtol':1e-2})
# -> see the minimizing values for f (the x vector), these are the same values for betas




"""
Fitting ODEs with the Levenberg–Marquardt algorithm
http://people.duke.edu/~ccc14/sta-663/CalibratingODEs.html
"""

# <-------------------------------------------------------------------------------------------

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





















