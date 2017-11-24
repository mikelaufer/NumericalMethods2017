import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.linalg import *
from numba import jit, prange
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# @jit
# def l2normdiff(phinew, phiold, dx2, dy2):
#     phi_diff = phinew - phiold
#     Rk = -((2/dx2) + (2/dy2))*phi_diff[1:-1,1:-1] + (1/dx2)*phi_diff[1:-1,2:] + (1/dx2)*phi_diff[1:-1,0:-2]  + (1/dy2)*phi_diff[2:,1:-1] + (1/dy2)*phi_diff[0:-2,1:-1] 
#     Rksquared = np.multiply(Rk,Rk)
#     return(math.sqrt(Rksquared.sum()))


@jit
def l2norm(phi, S, dx2, dy2):
    Rk =S[1:-1,1:-1] +((2/dx2) + (2/dy2))*phi[1:-1,1:-1] - (1/dx2)*phi[1:-1,2:] - (1/dx2)*phi[1:-1,0:-2]  - (1/dy2)*phi[2:,1:-1] - (1/dy2)*phi[0:-2,1:-1] 
    Rksquared = np.multiply(Rk,Rk)
    return (math.sqrt(Rksquared.sum()))

@jit
def l2normMSD(phi, S, dx2, dy2):
    ny, nx = phi.shape
    Rk = np.zeros((ny,nx))
    Rk[1:-1,1:-1] = S[1:-1,1:-1] +((2/dx2) + (2/dy2))*phi[1:-1,1:-1] - (1/dx2)*phi[1:-1,2:] - (1/dx2)*phi[1:-1,0:-2]  - (1/dy2)*phi[2:,1:-1] - (1/dy2)*phi[0:-2,1:-1] 
    Rksquared = np.multiply(Rk,Rk)
    R2sum = Rksquared.sum()
    norm = (math.sqrt(Rksquared.sum()))
    return (R2sum, Rk, norm)
@jit
def jacobistep(phi,S, dx2, dx):
    phin = phi.copy()
    phi[1:-1,1:-1] = (S[1:-1,1:-1] - (1/dy2)*phin[0:-2,1:-1] - (1/dy2)*phin[2:,1:-1] - (1/dx2)*phin[1:-1,0:-2] - (1/dx2)*phin[1:-1,2:])/(-((2/dx2) + (2/dy2)))
    return(phi)

@jit
def gaussstep(phi,S, dx2, dx):
    phin = phi.copy()
    nx, ny = phi.shape
    phin = np.copy(phi)
    for j in range(1, ny-1):
        for i in range(1,nx-1):
            phin[j,i] = (S[j,i] - (1/dy2)*phin[j-1,i] - (1/dy2)*phin[j+1,i] - (1/dx2)*phin[j,i-1] - (1/dx2)*phin[j,i+1])/(-((2/dx2) + (2/dy2)))
    return(phin)
@jit
def MSDstep(phi, S, R, R2sum, dx2, dy2):
    nx,ny = phi.shape
    phin = phi.copy()
    Sn = S.copy()
    Rn = R.copy()
    c = np.zeros((ny,nx))
    
    c[1:-1,1:-1] = -((2/dx2) + (2/dy2))*Rn[1:-1,1:-1] + (1/dx2)*Rn[1:-1,2:] + (1/dx2)*Rn[1:-1,0:-2]  + (1/dy2)*Rn[2:,1:-1] + (1/dy2)*Rn[0:-2,1:-1]
    rtc = np.sum(np.multiply(R, c))
    alpha = R2sum/rtc
    return( phin + alpha*R)

@jit
def CGstep(phi, S, R, R2sum, D, dx2, dy2):
    nx,ny = phi.shape
    phin = phi.copy()
    c = np.zeros((ny,nx))

    c[1:-1,1:-1] = -((2/dx2) + (2/dy2))*D[1:-1,1:-1] + (1/dx2)*D[1:-1,2:] + (1/dx2)*D[1:-1,0:-2]  + (1/dy2)*D[2:,1:-1] + (1/dy2)*D[0:-2,1:-1]
    rtc = np.sum(np.multiply(D, c))
    alpha = R2sum/rtc
    phin = phin + alpha*D
    R2 = math.sqrt(R2sum)
    R2sum2, Rk2, R22 = l2normMSD(phin, S, dx2, dy2)
    beta = (R2sum2)/(R2sum)
    D = Rk2 + beta*D
    return(phin, D)

def CGSstep(phi, S, R, R2sum, Rzero, D, Dstar, dx2, dy2):
    nx,ny = phi.shape
    phin = phi.copy()
    c = np.zeros((ny,nx))

    c[1:-1,1:-1] = -((2/dx2) + (2/dy2))*D[1:-1,1:-1] + (1/dx2)*D[1:-1,2:] + (1/dx2)*D[1:-1,0:-2]  + (1/dy2)*D[2:,1:-1] + (1/dy2)*D[0:-2,1:-1]
    rtc = np.sum(np.multiply(Rzero, c))
    alpha = np.sum(np.multiply(Rzero,R))/rtc
    G = Dstar -alpha*c
    phin = phin + alpha*(Dstar+G)
    R2sum2, Rk2, R22 = l2normMSD(phin, S, dx2, dy2)
    beta = np.sum(np.multiply(Rzero,Rk2))/np.sum(np.multiply(Rzero,R))
    Dstar = Rk2 + beta*G
    D = Dstar + beta*(G + beta*D)
    return(phin, D, Dstar)


if __name__ == "__main__":
    nx = 81
    ny = 81
    dx = 1./(nx-1)
    dx2 = dx**2
    dy = 1./(ny-1)
    dy2 = dy**2

    epsilon = 10e-7
    maxiters = 100000
    
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y, sparse=True)

    # Expressions
    phi_analytical = 500*np.exp(-50*((1-xx)**2 +yy**2)) + 100*xx*(1-yy)
    S = 50000*np.exp(-50*((1-xx)**2 + yy**2))*(100*((1-xx)**2 + yy**2) -2)
    phi_right = 100*(1-y) + 500*np.exp(-50*(y**2))
    phi_left = 500*np.exp(-50*(1+y**2))
    phi_bottom = 100*x + 500*np.exp(-50*((1-x)**2))
    phi_top = 500*np.exp(-50*((1-x)**2 +1)) 
    
    phi = np.zeros((ny,nx), dtype=float)
    phi[0,:] = phi_bottom
    phi[ny-1,:] = phi_top
    phi[:, 0] = phi_left
    phi[:, nx-1] = phi_right
    phistart = phi.copy()


    # Jacobi solve
    phiold = phi.copy()
    l2norm_phi = np.zeros(maxiters)
    for iteration in range(maxiters):
        phi = jacobistep(phi, S, dx2, dy2)
        l2norm_phi[iteration] = l2norm(phi, S, dx2, dy2)
        # l2norm_jacobi[iteration] = l2normorig(phi, S, dx2, dy2)
        if l2norm_phi[iteration] < epsilon:
            break
    phi_jacobi = phi.copy()
    l2norm_jacobi = l2norm_phi.copy()

     # Gauss-Seidel solve
    phi = np.copy(phistart)
    phiold = np.copy(phistart)
    l2norm_phi = np.zeros(maxiters)
    for iteration in range(maxiters):
        phi = gaussstep(phi, S, dx2, dy2)
        l2norm_phi[iteration] = l2norm(phi, S, dx2, dy2)
        if l2norm_phi[iteration] < epsilon:
            break
    phi_gauss = phi.copy()
    l2norm_gauss = l2norm_phi.copy()

    # MSD solve
    phi = np.copy(phistart)
    phiold = np.copy(phistart)
    l2norm_phi = np.zeros(maxiters)
    R2sum, R, l2norm_phi[0] = l2normMSD(phi, S, dx2, dy2)
    for iteration in range(1,maxiters):
        phi = MSDstep(phi, S, R, R2sum,  dx2, dy2)
        R2sum, R, l2norm_phi[iteration] = l2normMSD(phi, S, dx2, dy2)
        if l2norm_phi[iteration] < epsilon:
            break
    phi_MSD = phi.copy()
    l2norm_MSD = l2norm_phi.copy()

    # CG solve
    phi = np.copy(phistart)
    phiold = np.copy(phistart)
    l2norm_phi = np.zeros(maxiters)
    R2sum, R, l2norm_phi[0] = l2normMSD(phi, S, dx2, dy2)
    D = R.copy()
    for iteration in range(1,maxiters):
        phi, D = CGstep(phi, S, R, R2sum, D, dx2, dy2)
        R2sum, R, l2norm_phi[iteration] = l2normMSD(phi, S, dx2, dy2)
        if l2norm_phi[iteration] < epsilon:
            break
    phi_CG = phi.copy()
    l2norm_CG = l2norm_phi.copy()

    # # CGS solve
    phi = np.copy(phistart)
    phiold = np.copy(phistart)
    l2norm_phi = np.zeros(maxiters)
    R2sum, R, l2norm_phi[0] = l2normMSD(phi, S, dx2, dy2)
    Rzero = R.copy()
    D = R.copy()
    Dstar = R.copy()
    
    for iteration in range(1,maxiters):
        phi, D, Dstar = CGSstep(phi, S, R, R2sum, Rzero, D, Dstar, dx2, dy2)
        R2sum, R, l2norm_phi[iteration] = l2normMSD(phi, S, dx2, dy2)
        if l2norm_phi[iteration] < epsilon:
            break
    phi_CGS = phi.copy()
    l2norm_CGS = l2norm_phi.copy()
   

   


    # plt.figure(1)
    # plt.subplot(121)
    # plt.contourf(x,y,phi_CGS)
    # plt.colorbar()
    # plt.title('4th Order FD - Tri-diag')
    # plt.subplot(122)
    # plt.contourf(x,y,np.abs(phi_analytical-phi_CGS))
    # # plt.contourf(x,y,phi_gauss)
    # plt.colorbar()
    # plt.title('Numerical-Analytical Absolute Error')
    # plt.show()


    # plt.figure(2)
    # plt.subplot(121)
    # plt.contourf(x,y,phi_penta)
    # plt.colorbar()
    # plt.title('4th Order FD - Penta-diag')
    # plt.subplot(122)
    # plt.contourf(x,y,np.abs(phi_analytical-phi_penta))
    # plt.colorbar()
    # plt.title('Numerical-Analytical Absolute Error')
    
    plt.figure(3)
    plt.semilogy(np.arange(len(l2norm_jacobi)), l2norm_jacobi, label="Jacobi")
    plt.semilogy(np.arange(len(l2norm_gauss)), l2norm_gauss, label="Gauss-Seidel")
    plt.semilogy(np.arange(len(l2norm_MSD)), l2norm_MSD, label="MSD")
    plt.semilogy(np.arange(len(l2norm_CG)), l2norm_CG, label="CG")
    plt.semilogy(np.arange(len(l2norm_CGS)), l2norm_CGS, label="CGS")
    plt.xlim((-1000,25000))
    #plt.ylim((0,10**6))
    plt.xlabel("Iterations")
    plt.ylabel("Residual, R2")
    plt.legend()
    plt.grid(True)
    plt.show()

    # fig = plt.figure(figsize=(11, 7), dpi=100)
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(xx, yy, phi_analytical, cmap=cm.viridis, rstride=2, cstride=2)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.title('2nd Order Poisson')
    # plt.show()
