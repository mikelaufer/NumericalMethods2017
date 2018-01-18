import numpy as np
import gmsh_reader as gr

from scipy.sparse import csr_matrix, lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

import sys

try:
    fname = sys.argv[1]
    msh = gr.GmshReader(fname)
except:
    msh = gr.GmshReader('circle.msh')

Nelements = msh.nelements
Npoints   = msh.npoints
Nfaces    = msh.faces.size
NBfaces   = msh.bfaces.size

# all connectivity info is computed here
A = lil_matrix((Nelements, Nelements))
b = np.zeros(Nelements)

# initial value in all cells
phi      = np.zeros(Nelements)
phi_vert = np.zeros(Npoints)
tolerance = 1e-12


print
print('{:<30}'.format('Setting Cell Diffusion Coefficients...'))
k_elements = 400*np.ones(Nelements)
for elem_index in range(msh.nelements):
    if msh.cents[0]< 0.1 and msh.cents[0] > -0.1:
        if mesh.cents[1]< 0.1 and msh.cents[1] > -0.1:
            k_elements[elem_index] = 10000.and

print
print('{:<30}'.format('Computing Face Diffusion Coefficients...'))
k_faces = 400*np.ones(Nfaces)
for face_index in range(msh.faces):
    if face_index not in msh.bfaces:
        element_index_1 = msh.f2e[face_index,0]
        k_element_1 =  k_elements[element_index_1]
        for tmp_index in range(msh.nelements):
            if msh.e2f[element_index_1, tmp_index] == face_index:
                face_index_1 = tmp_index
                break
        dist_element_1 = msh.elinks[element_index_1, tmp_index, 0]

        element_index_2 = msh.f2e[face_index,1]
        k_element_2 =  k_elements[element_index_2]
        for tmp_index in range(msh.nelements):
            if msh.e2f[element_index_2, tmp_index] == face_index:
                face_index_2 = tmp_index
                break
        dist_element_2 = msh.elinks[element_index_2, tmp_index, 0]
        
        k_faces[face_index] = (k_element_1*k_element_2)/((k_element_1*dist_element_2 + k_element_2*dist_element_1)/(dist_element_1 + dist_element_2))

converged = False
iteration = 0
norm = -1
max_iter = 100
while not converged:
    iteration += 1
    
    # interpolate to the vertices
    phi_vert[:] = 0.0
    for i in range(msh.npoints):
        nelems = msh.Nv2e[i]

        interp_sum = 0.0
        for j in range(nelems):
            elem_index = msh.v2e[i,j]
            weight     = msh.idw_e2v[i, j]

            interp_sum += weight*phi[elem_index]
    
        phi_vert[i] = interp_sum

    A = lil_matrix((Nelements, Nelements))
    b[:]   = 0.0
        
    # assemble the coefficient and rhs matrix for all interior faces
    num_faces_done = 0
    for elem_index in range(msh.nelements):

        tmp1 = 0.0
        skew = 0.0
        for tmp_index in range(msh.elem_nfaces):
            nbr_index  = msh.neighbors[elem_index, tmp_index]

            if nbr_index != -1:
                num_faces_done += 1
                face_index = msh.e2f[elem_index, tmp_index]

                deltaf = msh.elinks[elem_index, tmp_index, 0]
                tdotI  = msh.elinks[elem_index, tmp_index, 1]
                ds     = msh.face_areas[face_index]

                node1, node2 = msh.xfaces[face_index]
                phi_diff = phi_vert[node2] - phi_vert[node1]

                tmp1                     += ds/deltaf              # Eq. 7.96a
                A[elem_index, nbr_index] = -ds/deltaf              # Eq. 7.96b
                skew                     += phi_diff/deltaf*tdotI  # Eq. 7.96c

        A[elem_index, elem_index] = tmp1
        b[elem_index] = -skew

    num_faces_done /= 2

    # assemble coefficient & rhs for all Dirichlet faces
    for tmp_face_index, face_index in enumerate(msh.dirichlet_faces):
        num_faces_done += 1
        elem_index      = msh.f2e[face_index, 0]
        elem_face_index = -1

        for tmp_index in range(msh.elem_nfaces):
            if msh.e2f[elem_index, tmp_index] == face_index:
                elem_face_index = tmp_index
                break
        
        deltaf = msh.elinks[elem_index, elem_face_index, 0]
        tdotI  = msh.elinks[elem_index, elem_face_index, 1]
        ds     = msh.face_areas[face_index]

        node1, node2 = msh.xfaces[face_index]

        b[elem_index] -= ( -msh.dirichlet_faces_val[tmp_face_index]*ds/deltaf + (phi_vert[node2]-phi_vert[node1])*tdotI/deltaf )

        A[elem_index, elem_index] += ds/deltaf

    # assemble coefficient & rhs for all Neumann faces
    for tmp_face_index, face_index in enumerate(msh.neumann_faces):
        num_faces_done += 1
        elem_index = msh.f2e[face_index, 0]
        ds         = msh.face_areas[face_index]

        b[elem_index] += msh.neumann_faces_val[tmp_face_index]*ds
        
    phi_new = spsolve(A.tocsc(), b)
    
    norm = np.linalg.norm(phi-phi_new)
    if (norm < tolerance or iteration >= max_iter):
        print('Converged in %d iterations'%iteration)
        converged = True
        break

    print('Iteration %d, Error = %g'%(iteration, norm))

    phi[:] = phi_new[:]

# write out solution
try:
    fname = sys.argv[2]
except:
    fname = 'phi_elliptic'
    pass

gr.write_vtk_ugrid(fname, msh, phi)
