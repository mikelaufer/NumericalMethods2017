import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # parameters
    nx = 52
    dx = 1 / (nx - 1)
    dt = 0.00001
    finaltime = 0.8

    x = np.linspace(0,1,nx)
    nt = int(finaltime/dt)
    dx2 = dx**2
    phi = np.ones(nx, dtype=float)
    
    for n in range(1,nt):
        phi_n = phi.copy()
        phi[1:-1] = phi_n[1:-1] + (dt/dx2)*(phi_n[2:] -2*phi_n[1:-1] + phi_n[0:-2])
        phi[0] = phi_n[1]
        phi[-1] = -dx*phi_n[-1] + phi_n[-2]

        if n*dt == 0.1:
            phi_01 = phi.copy()
        elif n*dt == 0.2:
            phi_02 = phi.copy()
        elif n*dt == 0.4:
            phi_04 = phi.copy()
        elif n*dt == 0.8:
            phi_08 = phi.copy()

    lamb = np.array([0.8603, 3.4256, 6.4373, 9.5293])
    Cn = (4*np.sin(lamb))/(2*lamb + np.sin(2*lamb))
    phi_anal = np.zeros((nt,nx))
    for n in range(nt):
        phi_anal[n] = Cn[0]*np.exp((-(lamb[0]**2))*n*dt)*np.cos(lamb[0]*x) +\
                      Cn[1]*np.exp((-(lamb[1]**2))*n*dt)*np.cos(lamb[1]*x) +\
                      Cn[2]*np.exp((-(lamb[2]**2))*n*dt)*np.cos(lamb[2]*x) +\
                      Cn[3]*np.exp((-(lamb[3]**2))*n*dt)*np.cos(lamb[3]*x)

    plt.plot(x, phi_01, label='Forward Euler 0.1')
    plt.plot(x, phi_anal[int(0.1/dt)], label='Analytical 0.1')
    plt.plot(x, phi_02, label='Forward Euler 0.2')
    plt.plot(x, phi_anal[int(0.2/dt)], label='Analytical 0.2')
    plt.plot(x, phi_04, label='Forward Euler 0.4')
    plt.plot(x, phi_anal[int(0.4/dt)], label='Analytical 0.4')
    
    
    
    plt.legend()
    plt.show()
    
    
        
    
    
    
