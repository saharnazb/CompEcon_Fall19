# -*- coding: utf-8 -*-
# used jupyter nbconvert
# Problem Set 8
@author: Saharnaz Babaei

import numpy as np
import math
from numba import jit, njit, autojit
import numba as nb
from scipy.optimize import fminbound
from scipy import interp
import scipy.integrate as integrate
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# to print plots inline
get_ipython().run_line_magic('matplotlib', 'inline')

'''
My sketch for the solution:
    - Set parameter values
    - Create Grid Space (For k, d, q(AR))
    - VFI and policy function definitions
    - Extract decision rules from solution
    - Visualize output

References used:
    - https://www.hackerearth.com/practice/python/object-oriented-programming/\
      classes-and-objects-i/tutorial/
    - https://macroeconomics.github.io/pages/Computation.html
    - Guimaraes, Bernardo. "Optimal external debt and default." (2007). (used
         to add an equation for debt (d))
'''

# define initial values
A = 1
alpha = 0.3
delta = 0.1
r = 0.02
beta = 1/(1+r)
sigma = 3
q_bar = beta
gamma = 0.01
psi = 0.5

# Grid of values for state variable over which function will be approximated
gridmin, gridmax, gridsize = 0.1, 80.0, 500
grid = np.linspace(gridmin, gridmax**1e-1, gridsize)**10

# Grid for q
mu = 1/(1+0.02)
rho = 0.8
sigma_eps = 0.018
num_draws = 5000 # number of shocks to draw
eps = np.random.normal(0.0, sigma_eps, size=(num_draws))
# Compute q_grid
q = np.empty(num_draws)
q[0] = 0.0 + eps[0]
for i in range(1, num_draws):
    q[i] = abs(rho * q[i - 1] + (1 - rho) * mu + eps[i])
# Plot distribution of q
sns.kdeplot(np.array(q))
plt.show()
q.min()
# transition probs
N = 2
sigma_q = 0.018
q_cutoff = (sigma_q * norm.ppf(np.arange(N+1) / N)) + mu
q_grid = ((N * sigma_q * (norm.pdf((q_cutoff[:-1] - mu) / sigma_q)
                - norm.pdf((q_cutoff[1:] - mu) / sigma_q))) + mu)

def integrand(x, sigma_q, sigma_eps, rho, mu, q_j, q_jp1):
    val = (np.exp((-1 * ((x - mu) ** 2)) / (2 * (sigma_q ** 2)))
           * (norm.cdf((q_jp1 - (mu * (1 - rho)) - (rho * x)) / sigma_eps)
            - norm.cdf((q_j - (mu * (1 - rho)) - (rho * x)) / sigma_eps)))
    return val

pi = np.empty((N, N))
for i in range(N):
    for j in range(N):
        results = integrate.quad(integrand, q_cutoff[i], q_cutoff[i + 1],
                    args = (sigma_q, sigma_eps, rho, mu,
                        q_cutoff[j], q_cutoff[j + 1]))
        pi[i,j] = (N / np.sqrt(2 * np.pi * sigma_q ** 2)) * results[0]
# convert log of shocks
q_qrid = [np.exp(x) for x in q_grid]
print('Transition matrix = ', pi)
print('pi sums = ', pi.sum(axis=0), pi.sum(axis=1))

# Grid for d
lbd, ubd, sized = 0.0, 80.0, 1001
d_grid = np.linspace(lbd, ubd**1e-1, sized)**10

# Parameters for the optimization
count=0
maxiter=10000
tol=1e-6
print('tol=%f' % tol)
print(grid.shape)

# Interpolation functions Class
class LinInterp:
    'Provides linear interpolation in one dimension.'

    def __init__(self, X, Y):
        '''
        Params: X and Y (sequences or arrays of interpolation points)
        '''
        self.X, self.Y = X, Y

    def __call__(self, z):
        '''Params: z is a number, sequence or array.
        This method makes a LinInterp callable.
        '''
        if isinstance(z, int) or isinstance(z, float):
            return interp ([z], self.X, self.Y)[0]
        else:
            return interp(z, self.X, self.Y)

# Utility, production growth and required functions definition
@nb.jit('f8[:](f8[:],f8)') #f8 is equivalent to float64
def U(c,sigma):
    if sigma!=1:
        u = Unb(c,sigma) #(c**(1-sigma)-1)/(1-sigma)
    else:
        u = np.log(c)
    return u

@nb.vectorize('f8(f8,f8)')
def Unb(c,sigma):
    if sigma!=1:
        u = (c**(1-sigma) - 1)/(1 - sigma)
    else:
        u = math.log(c)
    return u

@nb.vectorize('f8(f8,f8,f8)')
def F_nb(k,alpha,A):
    '''
    Cobb-Douglas production function
    F(k) = A * k^alpha
    '''
    return A * k**alpha

# Convert utility to use with minimization methods
def Utrf(kp,kmax,k,sigma,w):
    return -U(kmax + (1-delta) * k - kp,sigma)-beta*w(kp)

# Bellman and policy functions
@nb.jit('f8[:](f8[:],f8[:])')
def bellmannb(x,w0):
    '''
    Params: w (a LinInterp callable object which acts pointwise on arrays
            and is defined on state space).
    '''
    w=LinInterp(x,w0)
    vals = np.array([])
    for k in x:
        kmax=F_nb(k,alpha,A)
        vals=np.append(vals,-Utrf(fminbound(Utrf, 0, kmax,args=[kmax,k,sigma,w]),kmax,k,sigma,w))
    return vals

@nb.jit('f8[:](f8[:],f8[:])')
def policynb(x,w0):
    '''
    Parames: w (a LinInterp callable object which acts pointwise on arrays).
    According to references, q is not affecting k path but it effects d path.
    '''
    w=LinInterp(x,w0)
    vals = np.array([])
    fk = np.array([])
    d = np.array([0.8,])
    for k in x:
        kmax=F_nb(k,alpha,A)
        fk = np.append(fk, kmax)
        vals=np.append(vals,fminbound(Utrf, 0, kmax,args=[kmax,k,sigma,w]))
    for q in q_grid:
        for f in fk:
            #d = np.append(d, d=0)
            d = np.append(d, (1/q)  * (d[i-1] - gamma * f))
    return [vals, fk, d]

@nb.jit('f8(f8[:],f8[:])')
def errnb(V0,V1):
    maxi=0.0
    for i in range(V0.shape[0]):
        m=abs(V1[i]-V0[i])
        if m>=maxi:
            maxi=m
    return maxi

@nb.jit('f8[:]()')
def solvenb():
    count=0.0
    V0=U(grid,sigma)
    while count<maxiter:
        V1=bellmannb(grid,V0)
        err=errnb(V1,V0)
        V0=V1
        count+=1
        if err<tol:
            print(count)
            break
    return V0

U0=U(grid,sigma)
U0=bellmannb(grid,U0)

_, fk, _ = policynb(grid,U0)
_, _, d = policynb(grid,U0)
cap, _, _=policynb(grid,U0)

get_ipython().run_line_magic('timeit', 'bellmannb(grid,U0)')
get_ipython().run_line_magic('timeit', 'policynb(grid,U0)')
get_ipython().run_line_magic('timeit', 'solvenb()')

# Visualization
plt.figure(figsize=(10,10))
ax = plt.gca()
plt.plot(grid,cap,label='capital')
plt.legend()
plt.savefig("capital.png")
plt.show()

plt.figure(figsize=(10,10))
ax = plt.gca()
plt.plot(grid,fk,label='fk')
plt.legend()
plt.savefig("fk.png")
plt.show()

plt.figure(figsize=(10,10))
ax = plt.gca()
plt.plot(d_grid,d,label='d')
plt.legend()
plt.savefig("debt.png")
plt.show()
