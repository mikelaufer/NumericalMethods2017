import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

def psi(i,x, dx):
    if x > (i+1)*dx or x < (i-1)*dx:
        return 0
    elif x < i*dx:
        return (x - (i-1)*dx)/dx
    elif:
        return ((i+1)*dx - x)/dx
    

def galerkin1d(nx):
    x = np.linspace(0,1,nx)
    dx = 1.0/(nx-1)
    K = (2/dx)*np.ones((nx,nx))
    np.fill_diagonal(K,-1/dx)
    F = (-1/dx)*(np.sin((x[1:-1])*(2*x[1:-1]) - x[:-2] -x[2:] -2) + np.sin(x[:-2]) + np.sin(x[2:]))
    a = np.concatenate([[0], solve(K[1:-1,1:-1],F), [1]])
    return a

nx = 5
x = np.linspace(0,1,nx)
a = galerkin1d(nx)

nxplot = 100
phi_galerkin = np.zeros(nxplot)
plot_x = np.linspace(0,1,nxplot)

for i,xval in enumerate(plot_x):
    lower = int(xval)
    upper = int(xval)+1
    if xval - lower >= 0.5:
        phi_galerkin[i] = 
        
    
phi_galerkin = np.zeros(nx)

    
phi_galerkin = x -
basis = 

phi_analy = -np.cos(x) + (1+np.cos(1))*x +1

plt.plot(x,phi_analy)
plt.show()
    
