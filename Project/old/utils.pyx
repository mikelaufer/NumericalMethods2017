import numpy as np
cimport numpy as np

cimport cython
from cpython.list cimport PyList_GetItem, PyList_SetItem, PyList_GET_ITEM

@cython.boundscheck(False)
@cython.wraparound(True)
@cython.cdivision(True)
cpdef int get_face_neighbor(
    int dim, str etype, int elem_nverts, int elem_nfaces, int face_nnodes,
    int nnbrs,
    np.ndarray[ndim=1, dtype=np.int64_t] nbrs,
    np.ndarray[ndim=1, dtype=np.float64_t] face_centroid,
    np.ndarray[ndim=2, dtype=np.int64_t] e2v,
    np.ndarray[ndim=2, dtype=np.float64_t] points):
            
    # locals
    cdef int nbr, i, j
    cdef int face_neighbor = -1

    cdef np.ndarray[ndim=1, dtype=np.int64_t]   nbr_verts = np.zeros(elem_nverts, int)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] nbr_face_centroid = np.zeros(3)

    cdef bint nbr_found = False

    cdef int face_node_number
    cdef double x, y, z, diff2
    cdef list nbr_faces
    cdef tuple face

    cdef double factor = 1./face_nnodes

    for j in range(nnbrs):
        nbr = nbrs[j]

        for i in range(elem_nverts):
            nbr_verts[i] = e2v[nbr, i]
    
        nbr_faces = round_trip_connect(nbr_verts, dim, etype)
            
        for face_index in range(elem_nfaces):
            for i in range(3):
                nbr_face_centroid[i] = 0.0

            #face = nbr_faces[face_index]
            face = <tuple>PyList_GET_ITEM(nbr_faces, face_index)

            for i in range(face_nnodes):
                #x = points[nbr_faces[face_index][i], 0]
                #y = points[nbr_faces[face_index][i], 1]
                #z = points[nbr_faces[face_index][i], 2]

                #face_node_number = <int>PyList_GET_ITEM(face, i)

                face_node_number = face[i]
                x = points[ face_node_number, 0]
                y = points[ face_node_number, 1]
                z = points[ face_node_number, 2]

                nbr_face_centroid[0] = nbr_face_centroid[0] + factor*x
                nbr_face_centroid[1] = nbr_face_centroid[1] + factor*y
                nbr_face_centroid[2] = nbr_face_centroid[2] + factor*z

                #nbr_face_centroid += factor*points[ nbr_faces[face_index][i] ]
                    
            diff2 = (face_centroid[0]-nbr_face_centroid[0])**2 + \
                    (face_centroid[1]-nbr_face_centroid[1])**2 + \
                    (face_centroid[2]-nbr_face_centroid[2])**2
                
            if (diff2 < 1e-14):
                nbr_found     = True
                face_neighbor = nbr
                break

        if nbr_found : break
            
    return face_neighbor

@cython.boundscheck(False)
@cython.wraparound(True)
@cython.cdivision(True)
cpdef round_trip_connect(np.ndarray[ndim=1, dtype=np.int64_t] seq, 
                        int dim=2, str etype='tri'):

    cdef int i
    cdef list result = []

    if dim == 2:
        for i in range(len(seq)):
              result.append((seq[i], seq[(i+1)%len(seq)]))
    else:
        if etype == "hex":
            result = [ (seq[4], seq[0], seq[1], seq[5]), 
                       (seq[1], seq[2], seq[6], seq[5]),
                       (seq[3], seq[7], seq[6], seq[2]),
                       (seq[3], seq[0], seq[4], seq[7]),
                       (seq[0], seq[3], seq[2], seq[1]),
                       (seq[4], seq[5], seq[6], seq[7]) ]
                  
        elif etype == 'tet':
            result = [ (seq[0], seq[1], seq[2]),
                       (seq[3], seq[0], seq[1]),
                       (seq[2], seq[3], seq[0]),
                       (seq[1], seq[2], seq[3]), ]
        else:
            raise ValueError('round_trip_connect: Element type %s not supported yet'%etype)

    return result
