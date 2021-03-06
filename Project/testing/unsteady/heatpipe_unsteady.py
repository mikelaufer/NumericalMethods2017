import sys
import numpy as np
import gmsh_reader as gr
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from numba import jit, njit, prange
import time


#  Parameters
k_nom = 400.0
rhoc_nom = 8940.0*376.8
T_inf = 22.0  # Ambient temp
h = 100.0  # Convection coef
heat_flux = (40.0/(0.03**2))
meshname = 'heatpipe1.msh'
fname = 'phi_heatpipe' # VTU output file name
tolerance = 1e-7
max_iter = 3
final_time = 800.0 #sec
dt = 0.2 # sec


msh = gr.GmshReader(meshname)
Nfaces = msh.faces.size
NBfaces = msh.bfaces.size
npoints = msh.npoints
Nv2e = msh.Nv2e
v2e = msh.v2e
idw_e2v = msh.idw_e2v
elem_nfaces = msh.elem_nfaces
nelements = msh.nelements
e2f = msh.e2f
f2e = msh.f2e
elinks = msh.elinks
face_areas = msh.face_areas
neighbors = msh.neighbors
xfaces = msh.xfaces
dirichlet_faces = msh.dirichlet_faces
dirichlet_faces_val = msh.dirichlet_faces_val
neumann_faces = msh.neumann_faces
robin_faces = msh.robin_faces
adiabatic_faces = msh.adiabatic_faces
volumes = msh.volumes

# Convection (Robin BC) constants
alpha = -k_nom
beta = -h
gamma = -h*T_inf
gamma_div_beta = gamma/beta
alpha_div_beta = alpha/beta

# initialize variables
b = np.zeros(nelements)
phi = np.ones(nelements)*T_inf
phi_old = np.ones(nelements)*T_inf
phi_vert = np.zeros(npoints)



print
print('{:<30}'.format('Setting Cell Material Coefficients...'))
k_elements = k_nom*np.ones(nelements)
rhoc_elements = rhoc_nom*np.ones(nelements)
for elem_index in range(nelements):
    if msh.cents[elem_index][1] > 0.005 and msh.cents[elem_index][1] < 0.03:
            k_elements[elem_index] = 10000.0
            rhoc_elements[elem_index] = rhoc_nom


print
print('{:<30}'.format('Computing Face Diffusion Coefficients...\n'))
k_faces = 400*np.ones(msh.nfaces)
for face_index in range(msh.nfaces):
    if face_index not in msh.bfaces:
        element_index_1 = msh.f2e[face_index,0]
        k_element_1 =  k_elements[element_index_1]
        for tmp_index in range(msh.nelements):
            if msh.e2f[element_index_1, tmp_index] == face_index:
                face_index_1 = tmp_index
                break
        dist_element_1 = msh.elinks[element_index_1, face_index_1, 0]

        element_index_2 = msh.f2e[face_index,1]
        k_element_2 =  k_elements[element_index_2]
        for tmp_index in range(msh.nelements):
            if msh.e2f[element_index_2, tmp_index] == face_index:
                 face_index_2 = tmp_index
                 break
        dist_element_2 = msh.elinks[element_index_2, face_index_2, 0]
        
        k_faces[face_index] = (k_element_1*k_element_2)/((k_element_1*dist_element_2 + k_element_2*dist_element_1)/(dist_element_1 + dist_element_2))
        # if k_faces[face_index] > 500.0 and  k_faces[face_index] < 950.0:
        #     print (k_faces[face_index])

#@njit(nogil=True, parallel=True)
def vertices_interpolate(phi_vert, phi):  # interpolate to the vertices
    phi_vert[:] = 0.0
    #for i in prange(npoints):
    for i in range(npoints):
        nelems = Nv2e[i]
        interp_sum = 0.0
        for j in range(nelems):
            elem_index = v2e[i,j]
            weight     = idw_e2v[i, j]

            interp_sum += weight*phi[elem_index]
        phi_vert[i] = interp_sum
    return phi_vert

#@njit(nogil=True, parallel=True)
#@njit
def compute_interior_faces(A, b):
    #for elem_index in prange(nelements):
    for elem_index in range(nelements):
        tmp1 = 0.0
        skew = 0.0
        for tmp_index in range(elem_nfaces):
            nbr_index  = neighbors[elem_index, tmp_index]

            if nbr_index != -1:
                face_index = e2f[elem_index, tmp_index]

                deltaf = elinks[elem_index, tmp_index, 0]
                tdotI  = elinks[elem_index, tmp_index, 1]
                ds     = face_areas[face_index]

                node1, node2 = xfaces[face_index]
                phi_diff = phi_vert[node2] - phi_vert[node1]

                tmp1                     += k_faces[face_index]*ds/deltaf              # Eq. 7.96a
                A[elem_index, nbr_index] = -k_faces[face_index]*ds/deltaf              # Eq. 7.96b
                skew                     += k_faces[face_index]*phi_diff/deltaf*tdotI  # Eq. 7.96c
        time_A = rhoc_elements[elem_index]*volumes[elem_index]/dt
        time_b = (rhoc_elements[elem_index]*volumes[elem_index]/dt)*phi_old[elem_index]
        A[elem_index, elem_index] = tmp1 + time_A
        b[elem_index] = -skew + time_b
    return (A, b)

#@njit
def compute_neumann_faces(b):
    # assemble coefficient & rhs for all Neumann faces
    for tmp_face_index, face_index in enumerate(neumann_faces):
        elem_index = f2e[face_index, 0]
        ds         = face_areas[face_index]
        b[elem_index] += heat_flux*ds
    return(b)

#@jit
def compute_robin_faces(A, b):
    # assemble coefficient & rhs for all Robin faces
    for tmp_face_index, face_index in enumerate(robin_faces):
        elem_index      = f2e[face_index, 0]
        elem_face_index = -1

        for tmp_index in range(elem_nfaces):
            if e2f[elem_index, tmp_index] == face_index:
                elem_face_index = tmp_index
                break
        
        deltaf = elinks[elem_index, elem_face_index, 0]
        tdotI  = elinks[elem_index, elem_face_index, 1]
        ds     = face_areas[face_index]

        node1, node2 = xfaces[face_index]

        b[elem_index] -= ds*k_faces[face_index]*(-gamma_div_beta/(deltaf + alpha_div_beta) + (phi_vert[node2]-phi_vert[node1])*tdotI/(ds*(deltaf + alpha_div_beta)))

        A[elem_index, elem_index] += k_faces[face_index]*ds/(deltaf + alpha_div_beta)
    return(A, b)

#@njit
def compute_adiabatic_faces(b):
    for tmp_face_index, face_index in enumerate(adiabatic_faces):
        elem_index = f2e[face_index, 0]
        ds         = face_areas[face_index]

        b[elem_index] += 0
    return(b)

if __name__ == '__main__':
    t = 0.0
    timesteps = 0
    t0 = time.time()
    while t < final_time:
        timesteps += 1
        iteration = 0
        t += dt
        norm = -1
        phi_old[:] = phi[:]
        
        converged = False
        print('\nTime: %f'%(t))
                        
        while not converged:
            iteration += 1
            A = lil_matrix((nelements, nelements))
            b[:]   = 0.0
    
            phi_vert = vertices_interpolate(phi_vert, phi)   # interpolate to vertices

            A, b = compute_interior_faces(A, b)  # assemble the coefficient and rhs matrix for allfaces
            A, b = compute_robin_faces(A, b)
            b = compute_neumann_faces(b)
            b = compute_adiabatic_faces(b)
        
            phi_new = spsolve(A.tocsc(), b)
            norm = np.linalg.norm(phi-phi_new)
            print('Iteration %d, Error = %g'%( iteration, norm))
            if (norm < tolerance or iteration >= max_iter):
                converged = True
            phi[:] = phi_new[:]

        gr.write_vtk_ugrid(fname + '_' + str(timesteps), msh, phi)
    print(time.time()-t0)
    #gr.write_vtk_ugrid(fname + '_' + str(t), msh, phi)
