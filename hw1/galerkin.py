import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
import math
import seaborn as sns
sns.set_style("whitegrid")


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
    K = np.zeros((nx,nx))                   # Stiffness matrix
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
            
    F = np.zeros(nx)                         # Load vector
    F [0] = 0
    F[1:-1] = (-1.0/dx)*(2*np.cos(x[1:-1]) - np.cos(x[0:-2]) - np.cos(x[2:]))
    F[-1] = 1
    
    a = tridiagsolver(K,F)                   # Solve system
    
    nxplot = 200                             # Recombine phi from basis functions
    plot_x = np.linspace(0,1,nxplot)
    phi_galerkin = np.zeros(nxplot)
    for i in range(len(plot_x)):                    
        for j in range(len(a)):
            phi_galerkin[i] +=  a[j]*psi(j, plot_x[i], dx)
    return phi_galerkin

if __name__ == "__main__":
    plot_x = np.linspace(0,1,200)  # points for  plotting
    phi_galerkin5 = galerkin1d(nx=5)
    phi_galerkin9 = galerkin1d(nx=9)
    phi_analy = -np.cos(plot_x) + np.cos(1)*plot_x + 1

    plt.figure(1)
    plt.plot(plot_x, phi_analy, label= "Analytical")
    plt.plot(plot_x, phi_galerkin5, label="Galerkin FE")
    plt.title("Galerkin FE 5 Nodes")
    plt.ylabel("$\phi$")
    plt.xlabel("x")
    plt.legend()

    plt.figure(2)
    plt.plot(plot_x, phi_analy, label= "Analytical")
    plt.plot(plot_x, phi_galerkin9, label="Galerkin FE")
    plt.title("Galerkin FE 9 Nodes")
    plt.ylabel("$\phi$")
    plt.xlabel("x")
    plt.legend()

    plt.figure(3)
    plt.plot(plot_x, np.abs(phi_analy-phi_galerkin5), label= "5 Node num-analytical error")
    plt.plot(plot_x, np.abs(phi_analy-phi_galerkin9), label= "9 Node num-analytical error")
    plt.title("Galerkin FE Error Compared to Analytic Solution")
    plt.ylabel("Error")
    plt.xlabel("x")
    plt.legend()
    plt.show()
