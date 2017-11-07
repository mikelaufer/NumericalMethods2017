import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
import math


def tridiagsolver(K,F):
    ud = np.insert(np.diag(K,1), 0, 0)           # upper diagonal
    d = np.diag(K)                               # main diagonal
    ld = np.insert(np.diag(K,-1), len(F)-1, 0)   # lower diagonal
    ab = np.matrix([ud, d, ld])                  # simplified matrix
    a = solve_banded((1, 1), ab, F)
    return a

def psi(j,x, dx):
    if x > (j+1)*dx or x < (j-1)*dx:
        return 0
    elif x < j*dx:
        return (x - (j-1)*dx)/dx
    else:
        return ((j+1)*dx - x)/dx

def galerkin1d(nx):
    x = np.linspace(0,1,nx)
    dx = 1.0/(nx-1)
    K = np.zeros((nx,nx))
    for i in range(nx):
        if i == 0:
            K[i,i] = 1
            K[i,i+1] = 0
        elif i == len(K)-1:
            K[i,i] = 1
            K[i,i-1] = 0
        else:
            K[i,i] = 2/dx
            K[i,i-1] = -1/dx
            K[i,i+1] = -1/dx
    print("K", K)
    F = np.zeros(nx)
    for i in range(nx):
        if i == 0:
            F[i] = 0 
        elif i == nx - 1:
            F[i] = 1
        else:
            F[i] = (1.0/dx)*(2*np.exp(x[i]) - np.exp(x[i-1]) - np.exp(x[i+1]))
    print("F",F)
    a = tridiagsolver(K,F)
    nxplot = 200
    plot_x = np.linspace(0,1,nxplot)
    phi_galerkin = np.zeros(nxplot)
    for i in range(len(plot_x)):
        for j in range(len(a)):
            phi_galerkin[i] +=  a[j]*psi(j, plot_x[i], dx)
    fluxl = a[0]*(1/dx) + 0
    fluxr = a[-1]*(1/dx) + 0
    return phi_galerkin

if __name__ == "__main__":
    plot_x = np.linspace(0,1,200)  # points for  plotting
    phi_galerkin5 = galerkin1d(nx=5)
    phi_galerkin9 = galerkin1d(nx=9)
    phi_analy = np.exp(plot_x) + (2-math.exp(1))*plot_x - 1

    plt.figure(1)
    plt.plot(plot_x, phi_analy, label= "Analytical")
    plt.plot(plot_x, phi_galerkin5, label="Galerkin FE")
    plt.title("Galerkin FE 5 Nodes")
    plt.legend()

    plt.figure(2)
    plt.plot(plot_x, phi_analy, label= "Analytical")
    plt.plot(plot_x, phi_galerkin9, label="Galerkin FE")
    plt.title("Galerkin FE 9 Nodes")
    plt.legend()

    plt.figure(3)
    plt.plot(plot_x, np.abs(phi_analy-phi_galerkin5), label= "5 Node num-analytical error")
    plt.plot(plot_x, np.abs(phi_analy-phi_galerkin9), label= "9 Node num-analytical error")
    plt.title("Galerkin FE error compared to analytic")
    plt.legend()
    plt.show()
