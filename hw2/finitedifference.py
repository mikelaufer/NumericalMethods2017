import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

nx = 21
ny = 21
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xx, yy = np.meshgrid(x, y, sparse=True)

# Expressions
phi_analytical = 500*np.exp(-50*((1-xx)**2 + yy**2)) + 100*xx*(1-yy)
S = 50000*np.exp(-50*((1-xx)**2 + yy**2))*(100*((1-xx)**2 + yy**2) -2)
phi_right = 100*(1-y) + 500*np.exp(-50*(y**2))
phi_left = 500*np.exp(-50*(1+y**2))
phi_bottom = 100*(x) + 500*np.exp(-50*((1 - x)**2))
phi_top = 500*np.exp(-50*((1-x)**2 +1))

# Coef. Matrix, RHS Vector
A = np.zeros((nx*ny, nx*ny), dtype=float)
Q = np.zeros(nx*ny, dtype=float)
dx = 1.0/(nx - 1)
dy = 1.0/(ny - 1)
dx2 = dx*dx
dy2 = dy*dy

for i in range(1, nx-1):
    for j in range(1, ny-1):
        k = (j-1)*nx + i -1
        A[k,k] = -(2.0/dx2 + 2.0/dy2)
        A[k, k-1] = 1/dx2
        A[k, k+1] = 1/dx2
        A[k, k-nx] = 1/dy2
        A[k, k+nx] = 1/dy2
        Q[k] = S[j,i]

# Left Boundary
i = 0
for j in range(ny):
    k = (j-1)*nx + i -1
    A[k,k] = 1
    Q[k]  = phi_left[j]

# Right Boundary
i = nx - 1
for j in range(ny):
    k = (j-1)*nx + i -1
    A[k,k] = 1
    Q[k]  = phi_right[j]

# Bottom Boundary
j = 0
for i in range(nx):
    k = (j-1)*nx + i -1
    A[k,k] = 1
    Q[k]  = phi_bottom[i]

# Top Boundary
j = ny - 1
for i in range(nx):
    k = (j-1)*nx + i -1
    A[k,k] = 1
    Q[k]  = phi_top[i]

# Solve and unpack solution
phi2d = np.zeros((nx,ny))
phi1d = np.linalg.solve(A,Q)    
for i in range(nx):
    for j in range(ny):
        k = (j-1)*nx + i - 1
        phi2d[j,i] = phi1d[k]

# Absolute error
error = np.abs(phi2d - phi_analytical)

# 2D plot
plt.figure(1)
plt.subplot(121)
plt.contourf(x,y,phi2d)
plt.colorbar()
plt.title('Numerical Solution 21 Nodes')
plt.subplot(122)
plt.contourf(x,y,error)
plt.colorbar()
plt.title('Numerical-Analytical Error')
plt.title('Absolute Error')


# 3d plot
fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, phi2d, cmap=cm.viridis, rstride=2, cstride=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('FD Solution 3D')

plt.show()
