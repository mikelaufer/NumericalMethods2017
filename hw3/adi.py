import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.linalg import *
from numba import jit, prange
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#import seaborn as sns
#sns.set_style("whitegrid")

@jit
def trirowstepper(phi, S, phi_left, phi_right, phi_bottom, phi_top, dx2, dy2):
    nx, ny = phi.shape
    phin = np.copy(phi)
    d = np.zeros(nx, dtype=float)
    Q = np.zeros(nx,dtype=float)
    c = np.zeros(nx-1,dtype=float)
    a = np.zeros(nx-1,dtype=float)
    for j in range(1, ny-1):
        if j == 1 or j == ny - 2:
            for i in range(nx):
                if i == 0:
                    d[i] = 1.0
                    c[i] = 0.0
                    Q[i] = phi_left[j]
                elif i == nx-1:
                    d[i] = 1.0
                    a[i-1] = 0.0
                    Q[i] = phi_right[j]
                else:
                    d[i] = -(2/(dx2) + 2/(dy2))
                    c[i] = 1.0/(dx2)
                    a[i-1] = 1.0/(dx2)
                    Q[i] =  S[j,i] - (1.0/dy2)*(phin[j+1,i]) -(1.0/dy2)*(phi[j-1,i])
        else:
            for i in range(nx):
                if i == 0:
                    d[i] = 1.0
                    c[i] = 0.0
                    #Q[i] = phi_left[ny -1 - j]
                    Q[i] = phi_left[j]
                elif i == nx-1:
                    d[i] = 1.0
                    a[i-1] = 0.0
                    #Q[i] = phi_right[ny -1 - j]
                    Q[i] = phi_right[j]
                elif ((i == 1) or (i == nx-2)):
                    d[i] = -(2.0/(dx2) + 2.0/(dy2))
                    c[i] = 1.0/(dx2)
                    a[i-1] = 1.0/(dx2)
                    Q[i] =  S[j,i] - (1.0/dy2)*(phin[j+1,i]) -(1.0/dy2)*(phi[j-1,i])
                else:
                    d[i] = -(5.0/(2.0*dx2) + 5.0/(2.0*dy2))
                    c[i] = 4.0/(3.0*dx2)
                    a[i-1] = 4.0/(3.0*dx2)
                    Q[i] =  S[j,i] + (1.0/(12.0*dy2))*(phin[j+2,i] -4*phin[j+1,i] -4*phi[j-1,i] \
                            + phi[j-2,i]) -(1/dy2)*(phin[j+1,i] +phi[j-1,i]) +(1.0/(12*dx2))*(phin[j,i+2] + phi[j,i-2])
                    
        phi[j,:] = tridiag(d, c, a, Q)
    return phi

@jit
def tricolstepper(phi, S, phi_left, phi_right, phi_bottom, phi_top, dx2, dy2):
    nx, ny = phi.shape
    phin = np.copy(phi)
    d = np.zeros(ny, dtype=float)
    Q = np.zeros(ny,dtype=float)
    c = np.zeros(ny-1,dtype=float)
    a = np.zeros(ny-1,dtype=float)
    for i in range(1, nx-1):
        if i == 1 or i == nx - 2:
            for j in range(ny):
                if j == 0:
                    d[j] = 1.0
                    c[j] = 0.0
                    Q[j] = phi_bottom[i]
                elif j == ny-1:
                    d[j] = 1.0
                    a[j-1] = 0.0
                    Q[j] = phi_top[i]
                else:
                    d[j] = -(2/(dy2) + 2/(dx2))
                    c[j] = 1.0/(dy2)
                    a[j-1] = 1.0/(dy2)
                    Q[j] =  S[j,i] - (1.0/dx2)*(phin[j,i+1]) -(1.0/dx2)*(phi[j,i-1])
            
        else:
            for j in range(ny):
                if j == 0:
                    d[j] = 1.0
                    c[j] = 0.0
                    Q[j] = phi_bottom[i]
                elif j == ny-1:
                    d[j] = 1.0
                    a[j-1] = 0.0
                    Q[j] = phi_top[i]
                elif ((j == 1) or (j == ny-2)):
                    d[j] = -(2.0/(dy2) + 2.0/(dx2))
                    c[j] = 1.0/(dy2)
                    a[j-1] = 1.0/(dy2)
                    Q[j] =  S[j,i] - (1.0/dx2)*(phin[j,i+1]) -(1.0/dx2)*(phi[j,i-1])
                else:
                    d[j] = -(5.0/(2.0*dy2) + 5.0/(2.0*dx2))
                    c[j] = 4.0/(3.0*dy2)
                    a[j-1] = 4.0/(3.0*dy2)
                    Q[j] =  S[j,i] + (1.0/(12.0*dx2))*(phin[j,i+2] -4*phin[j,i+1] -4*phi[j,i-1] \
                            + phi[j,i-2]) -(1/dx2)*(phin[j,i+1] +phi[j,i-1]) +(1.0/(12*dy2))*(phin[j+2,i] + phin[j-2,i])
                    
        phi[:,i] = tridiag(d, c, a, Q)
    return phi


@jit
def pentrowstepper(phi, S, phi_left, phi_right, phi_bottom, phi_top, dx2, dy2):
    nx, ny = phi.shape
    phin = np.copy(phi)
    d = np.zeros(nx, dtype=float)
    Q = np.zeros(nx,dtype=float)
    c = np.zeros(nx-1,dtype=float)
    a = np.zeros(nx-1,dtype=float)
    f = np.zeros(nx-2,dtype=float)
    e = np.zeros(nx-2,dtype=float)
        
    
    for j in range(1, ny-1):
        if j == 1 or j == ny - 2:
            for i in range(nx):
                if i == 0:
                    d[i] = 1.0
                    c[i] = 0.0
                    f[i] = 0.0
                    Q[i] = phi_left[j]
                elif i == 1:
                    d[i] = -(2/(dx2) + 2/(dy2))
                    c[i] = 1.0/(dx2)
                    a[i-1] = 1.0/(dx2)
                    f[i] = 0.0
                    Q[i] =  S[j,i] - (1.0/dy2)*(phin[j+1,i]) -(1.0/dy2)*(phi[j-1,i])
            
                elif i == nx-1:
                    d[i] = 1.0
                    a[i-1] = 0.0
                    e[i-2] = 0
                    Q[i] = phi_right[j]

                elif i == nx-2:
                    d[i] = -(2/(dx2) + 2/(dy2))
                    c[i] = 1.0/(dx2)
                    a[i-1] = 1.0/(dx2)
                    e[i-2] = 0
                    Q[i] =  S[j,i] - (1.0/dy2)*(phin[j+1,i]) -(1.0/dy2)*(phi[j-1,i])
                                
                else:
                    d[i] = -(2/(dx2) + 2/(dy2))
                    c[i] = 1.0/(dx2)
                    a[i-1] = 1.0/(dx2)
                    e[i-2] = 0
                    f[i] = 0
                    Q[i] =  S[j,i] - (1.0/dy2)*(phin[j+1,i]) -(1.0/dy2)*(phi[j-1,i])
            phi[j,:] = tridiag(d, c, a, Q)
        else:
            for i in range(nx):
                if i == 0:
                    d[i] = 1.0
                    c[i] = 0.0
                    f[i] = 0.0
                    Q[i] = phi_left[j]
                elif i == 1:
                    d[i] = -(2/(dx2) + 2/(dy2))
                    c[i] = 1.0/(dx2)
                    a[i-1] = 1.0/(dx2)
                    f[i] = 0
                    Q[i] =  S[j,i] - (1.0/dy2)*(phin[j+1,i]) -(1.0/dy2)*(phi[j-1,i])
            
                elif i == nx-1 :
                    d[i] = 1.0
                    a[i-1] = 0.0
                    e[i-2] = 0
                    Q[i] = phi_right[j]
                elif i == nx-2:
                    d[i] = -(2/(dx2) + 2/(dy2))
                    c[i] = 1.0/(dx2)
                    a[i-1] = 1.0/(dx2)
                    e[i-2] = 0.0
                    Q[i] =  S[j,i] - (1.0/dy2)*(phin[j+1,i]) -(1.0/dy2)*(phi[j-1,i])
                          
                else:
                    d[i] = -(5.0/(2.0*dx2) + 5.0/(2.0*dy2))
                    c[i] = 4.0/(3.0*dx2)
                    a[i-1] = 4.0/(3.0*dx2)
                    f[i] = -(1.0/(12.0*dx2))
                    e[i-2] = -(1.0/(12.0*dx2))
                    Q[i] =  S[j,i] + (1.0/(12.0*dy2))*(phin[j+2,i] -4*phin[j+1,i] -4*phi[j-1,i] \
                            +phi[j-2,i]) -(1/dy2)*(phin[j+1,i] +phi[j-1,i])

            #print(d)
            #print(c)
            #print(f)
            phi[j,:] = pentadiag(d, f, c, a, e, Q)
    return phi

@jit
def pentcolstepper(phi, S, phi_left, phi_right, phi_bottom, phi_top, dx2, dy2):
    nx, ny = phi.shape
    phin = np.copy(phi)
    d = np.zeros(ny, dtype=float)
    Q = np.zeros(ny,dtype=float)
    c = np.zeros(ny-1,dtype=float)
    a = np.zeros(ny-1,dtype=float)
    f = np.zeros(ny-2,dtype=float)
    e = np.zeros(ny-2,dtype=float)
    
    for i in range(1, nx-1):
        if i == 1 or i == nx - 2:
            for j in range(ny):
                if j == 0:
                    d[j] = 1.0
                    c[j] = 0.0
                    f[j] = 0.0
                    Q[j] = phi_bottom[i]
                elif j == 1:
                    d[j] = -(2/(dx2) + 2/(dy2))
                    c[j] = 1.0/(dy2)
                    a[j-1] = 1.0/(dy2)
                    f[j] = 0.0
                    Q[j] =  S[j,i] - (1.0/dx2)*(phin[j,i+1]) -(1.0/dx2)*(phi[j,i-1])
            
                elif j == ny-1:
                    d[j] = 1.0
                    a[j-1] = 0.0
                    e[j-2] = 0
                    Q[j] = phi_top[i]

                elif j == ny-2:
                    d[j] = -(2/(dx2) + 2/(dy2))
                    c[j] = 1.0/(dy2)
                    a[j-1] = 1.0/(dy2)
                    e[j-2] = 0
                    Q[j] =  S[j,i] - (1.0/dx2)*(phin[j,i+1]) -(1.0/dx2)*(phi[j,i-1])
                                
                else:
                    d[j] = -(2/(dx2) + 2/(dy2))
                    c[j] = 1.0/(dy2)
                    a[j-1] = 1.0/(dy2)
                    e[j-2] = 0
                    f[j] = 0
                    Q[j] =  S[j,i] - (1.0/dx2)*(phin[j,i+1]) -(1.0/dx2)*(phi[j,i-1])
            phi[:,i] = tridiag(d, c, a, Q)
        else:
            for j in range(ny):
                if j == 0:
                    d[j] = 1.0
                    c[j] = 0.0
                    f[j] = 0.0
                    Q[j] = phi_bottom[i]
                elif j == 1:
                    d[j] = -(2/(dx2) + 2/(dy2))
                    c[j] = 1.0/(dy2)
                    a[j-1] = 1.0/(dy2)
                    f[j] = 0
                    Q[j] =  S[j,i] - (1.0/dx2)*(phin[j,i+1]) -(1.0/dx2)*(phi[j,i-1])
            
                elif j == ny-1:
                    d[j] = 1.0
                    a[j-1] = 0.0
                    e[j-2] = 0
                    Q[j] = phi_top[i]
                elif j == ny-2:
                    d[j] = -(2/(dx2) + 2/(dy2))
                    c[j] = 1.0/(dy2)
                    a[j-1] = 1.0/(dy2)
                    e[j-2] = 0.0
                    Q[j] =  S[j,i] - (1.0/dx2)*(phin[j,i+1]) -(1.0/dx2)*(phi[j,i-1])
                          
                else:
                    d[j] = -(5.0/(2.0*dx2) + 5.0/(2.0*dy2))
                    c[j] = 4.0/(3.0*dy2)
                    a[j-1] = 4.0/(3.0*dy2)
                    f[j] = -(1.0/(12.0*dy2))
                    e[j-2] = -(1.0/(12.0*dy2))
                    Q[j] =  S[j,i] + (1.0/(12.0*dx2))*(phin[j,i+2] -4*phin[j,i+1] \
                            -4*phi[j,i-1] +phi[j,i-2]) -(1/dx2)*(phin[j,i+1] +phi[j,i-1])
                    
            phi[:,i] = pentadiag(d, f, c, a, e, Q)
    return phi

@jit
def tridiag(d, c, a, Q):
    N = len(Q)
    ans = np.zeros(N)
    d = np.copy(d)
    c = np.copy(c)
    a = np.copy(a)
    Q = np.copy(Q)
    
    for i in range(1,N):
        const = a[i-1]/d[i-1]
        d[i] = d[i] - const*c[i-1]
        Q[i] = Q[i] - const*Q[i-1]
    ans[N-1] = Q[N-1]/d[N-1]
    for i in range(N-2, -1, -1):
        ans[i] = (Q[i] -c[i]*ans[i+1])/d[i]
    return ans

@jit
def pentadiag(d, f, c, a, e, Q):
    N = len(Q)
    ans = np.zeros(N)
    d = np.copy(d)
    f = np.copy(f)
    c = np.copy(c)
    a = np.copy(a)
    e = np.copy(e)
    Q = np.copy(Q)
    
    for i in range(1, N-1):
        const1 = a[i-1]/d[i-1]
        d[i] = d[i] -const1*c[i-1]
        c[i] = c[i] -const1*f[i-1]
        Q[i] = Q[i] -const1*Q[i-1]
        const2 = e[i-1]/d[i-1]
        a[i] = a[i] -const2*c[i-1]
        d[i+1] = d[i+1] -const2*f[i-1]
        Q[i+1] = Q[i+1] - const2*Q[i-1]
    const3 = a[N-2]/d[N-2]
    d[N-1] = d[N-1] -const3*c[N-2]
    ans[N-1] = (Q[N-1] -const3*Q[N-2])/d[N-1]
    ans[N-2] = (Q[N-2] -c[N-2]*Q[N-1])/d[N-2]
    for i in range(N-3, -1, -1):
        ans[i] = (Q[i] -c[i]*ans[i+1] -f[i]*ans[i+2])/d[i]
    return ans
    # c = np.concatenate([[0],c])
    # f = np.concatenate([[0,0],f])
    # a = np.concatenate([a,[0]])
    # e = np.concatenate([e,[0,0]])
    # ab = np.matrix([f,c, d, a, e])                  # simplified matrix
    # ans = solve_banded((2, 2), ab, Q)
    # return ans
@jit
def l2norm(phi, dx2, dy2):
   ny, nx = phi.shape
   Rk = phi[2:-2,2:-2]*(-(5.0/(2.0*dx2) + 5.0/(2.0*dy2))) + phi[2:-2,3:-1]*(4.0/(3.0*dx2)) \
        + phi[2:-2,1:-3]*(4.0/(3.0*dx2)) + phi[3:-1,2:-2]*(4.0/(3.0*dy2)) \
        + phi[1:-3,2:-2]*(4.0/(3.0*dy2)) + phi[2:-2,4:]*(-1.0/(12*dx2)) + \
        phi[2:-2,:-4]*(-1.0/(12*dx2)) + phi[4:,2:-2]*(-1.0/(12*dy2)) \
        + phi[:-4,2:-2]*(-1.0/(12*dy2))
   Rksquared = np.multiply(Rk,Rk)
   return math.sqrt(Rksquared.sum())


if __name__ == "__main__":
    nx = 81
    ny = 81
    dx = 1/(nx-1)
    dx2 = dx**2
    dy = 1/(ny-1)
    dy2 = dy**2

    epsilon = 10e-8
    maxiters = 8000
    
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y, sparse=True)

    # Expressions
    phi_analytical = ((xx -0.5)**2)*np.sinh(10*(xx-0.5)) \
                     + ((yy-0.5)**2)*np.sinh(10*(yy-0.5)) + np.exp(2*xx*yy)
    S = 2*np.sinh(10*(xx-0.5)) + 40*(xx-0.5)*np.cosh(10*(xx-0.5)) \
        + 100*((xx-0.5)**2)*np.sinh(10*(xx-0.5)) + 2*np.sinh(10*(yy-0.5)) \
        + 40*(yy-0.5)*np.cosh(10*(yy-0.5)) \
        + 100*((yy-0.5)**2)*np.sinh(10*(yy-0.5)) \
        + 4*(xx**2 + yy**2)*np.exp(2*xx*yy)
    phi_left = 0.25*np.sinh(-5) + ((y-0.5)**2)*np.sinh(10*(y-0.5)) + 1
    phi_right = 0.25*np.sinh(5) + ((y-0.5)**2)*np.sinh(10*(y-0.5)) + np.exp(2*y)
    phi_bottom = 0.25*np.sinh(-5) + ((x-0.5)**2)*np.sinh(10*(x-0.5)) + 1
    phi_top = 0.25*np.sinh(5) + ((x-0.5)**2)*np.sinh(10*(x-0.5)) + np.exp(2*x)

    
    phi = np.zeros((ny,nx), dtype=float)
    phi[0,:] = phi_bottom
    phi[ny-1,:] = phi_top
    phi[:, 0] = phi_left
    phi[:, nx-1] = phi_right
    phistart = phi.copy()


    # Tridiag solve
    phiold = phi.copy()
    l2norm_phi = np.zeros(maxiters)
    for iteration in range(maxiters):
        phi = trirowstepper(phi, S, phi_left, phi_right, phi_bottom, phi_top, dx2, dy2)
        phi = tricolstepper(phi, S, phi_left, phi_right, phi_bottom, phi_top, dx2, dy2)
        l2norm_phi[iteration] = l2norm(phi-phiold, dx2, dy2)
        if l2norm_phi[iteration] < epsilon:
            break
        phiold = phi.copy()
    phi_tri = phi.copy()
    l2norm_tri = l2norm_phi.copy()

    # Pentadiag solve
    phi = np.copy(phistart)
    phiold = np.copy(phistart)
    l2norm_phi = np.zeros(maxiters)
    for iteration in range(maxiters):
        phi = pentrowstepper(phi, S, phi_left, phi_right, phi_bottom, phi_top, dx2, dy2)
        phi = pentcolstepper(phi, S, phi_left, phi_right, phi_bottom, phi_top, dx2, dy2)
        l2norm_phi[iteration] = l2norm(phi-phiold, dx2, dy2)
        if l2norm_phi[iteration] < epsilon:
            break
        phiold = phi.copy()
    phi_penta = phi.copy()
    l2norm_penta = l2norm_phi.copy()

    plt.figure(1)
    plt.subplot(121)
    plt.contourf(x,y,phi_tri)
    plt.colorbar()
    plt.title('4th Order FD - Tri-diag')
    plt.subplot(122)
    plt.contourf(x,y,np.abs(phi_analytical-phi_tri))
    plt.colorbar()
    plt.title('Numerical-Analytical Absolute Error')

    plt.figure(2)
    plt.subplot(121)
    plt.contourf(x,y,phi_penta)
    plt.colorbar()
    plt.title('4th Order FD - Penta-diag')
    plt.subplot(122)
    plt.contourf(x,y,np.abs(phi_analytical-phi_penta))
    plt.colorbar()
    plt.title('Numerical-Analytical Absolute Error')
    
    plt.figure(3)
    plt.semilogy(np.arange(len(l2norm_tri))*2, l2norm_tri, label="Tri-diag")
    plt.semilogy(np.arange(len(l2norm_penta))*2, l2norm_penta, label="Penta-diag")
    plt.xlim((-100,8000))
    plt.ylim((0,10**6))
    plt.xlabel("Iterations")
    plt.ylabel("Residual, R2")
    plt.legend()
    plt.grid(True)
    plt.show()

    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, phi_tri, cmap=cm.viridis, rstride=2, cstride=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('4th Order FD - Tri-diag')

    

    plt.show()
