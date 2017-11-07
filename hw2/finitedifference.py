import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

nx = 6
ny = 6
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xx, yy = np.meshgrid(x, y, sparse=True)

psi_analytical = 500*np.exp(-50*((1-xx)**2 + yy**2)) + 100*xx*(1-yy)

S = 50000*np.exp(-50*((1-xx)**2 + yy**2))*(100*((1-xx)**2 + yy**2) -2)
psi_right = 100*(1-yy) + 500*np.exp(-50*(yy**2))
psi_left = 100*(1-yy) + 500*np.exp(-50*(yy**2))
psi_bottom = 100*(1-yy) + 500*np.exp(-50*(yy**2))
psi_top = 100*(1-yy) + 500*np.exp(-50*(yy**2))


A = np.zeros((nx*ny, nx*ny), dtype=float)
Q = np.zeros(nx*ny, dtype=float)
dx = 1.0/(nx - 1)
dy = 1.0/(ny - 1)
dx2 = dx*dx
dy2 = dy*dy

for i in range(1, nx-1):
    for j in range(1, ny-1):
        k = (j)*nx + i - 1
        A[k,k] = -(2.0/dx2 + 2.0/dy2)
        A[k, k-1] = 1/dx2
        A[k, k+1] = 1/dx2
        A[k, k-n] = 1/dy2
        A[k, k+n] = 1/dy2
        Q[k] = S[i,j]
        
        #print(A[k,k])
i = 0
for j in range(ny):
    k = (j)*nx + i - 1
    A[k,k] = 1
    Q[k]  = psi_left[j,i


# x = np.linspace(0, 1, nx)
# y = np.linspace(0, 1, ny)
# xx, yy = np.meshgrid(x, y, sparse=True)

# psi_analytical = 500*np.exp(-50*((1-xx)**2 + yy**2)) + 100*xx*(1-yy)

## 2D plot
# plt.contourf(x,y,psi_analytical)
# plt.colorbar()

# 3d plot
# fig = plt.figure(figsize=(11, 7), dpi=100)
# ax = fig.gca(projection='3d')
# ax.plot_surface(xx, yy, psi_analytical, cmap=cm.viridis, rstride=2, cstride=2)
plt.show()
