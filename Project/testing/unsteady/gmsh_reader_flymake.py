import numpy as np
import numpy.ma as ma
from scipy.spatial import cKDTree

import pyximport;
#pyximport.install()
pyximport.install(setup_args={'include_dirs': np.get_include()})
import utils

import matplotlib.pyplot as plt

class BBox:
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.xmin = xmin
        self.ymin = ymin

        self.xmax = xmax
        self.ymax = ymax

        self.zmin = zmin
        self.zmax = zmax

class GmshReader(object):
    def __init__(self, mshfile):
        self.mshfile = mshfile

        self.find_elem_type()

        print
        print('{:<30}'.format('Reading physical tags...'))
        self._read_physical_tags()
        print
        print('{:<30}'.format('Reading Nodes and elements...'))
        self.read_nodes_and_elements()
        print('{:^10}'.format('Number of elements:'))
        print('{:^10}'.format('Lines     = %d'%self.nlines))
        print('{:^10}'.format('Triangles = %d'%self.ntri))
        print('{:^10}'.format('Quads     = %d'%self.nquads))
        print('{:^10}'.format('Tets      = %d'%self.ntets))
        print('{:^10}'.format('Hexs      = %d'%self.nhexs))
        print('')

        # kd-tree description of the mesh
        print
        print('{:<30}'.format('Generating KDtree...'))
        self.kd_tree = cKDTree(self.cents)

        self.bbox = bbox = self.get_bbox()

        # get boundary tags
        btags = {}
        btree = {}
        for index, tag in enumerate(self.tags.keys()):
            tagdata = self.tags[tag]
            tagdim = tagdata[0]
            if tagdim == self.dim-1:
                lines      = self.get_bfaces(tag)
                line_cents = np.zeros(shape=(lines.shape[0], 3))

                for tmp, line in enumerate(lines):
                    line_cents[tmp] = np.sum(self.points[line], axis=0)/self.face_nnodes
                
                btags[tag] = self.get_bfaces(tag)
                btree[tag] = cKDTree(line_cents)
                
        self.btags = btags
        self.btree = btree

        print
        print('{:<30}'.format('Setting basic connectivity...'))
        self.setup_connectivity()
        self.e2v = self.element2vertices

        # vertex to element connectivity
        print
        print('{:<30}'.format('Setting vertex connectivity...'))
        self.get_vertex_connectivity()

        # element to face interpolation coefficients
        print
        print('{:<30}'.format('Setting IDW coefficients...'))
        self.get_e2f_idw_coefficients()

        # element volumes
        print
        print('{:<30}'.format('Setting element volumes...'))
        self.get_element_volumes()

        # element outward pointing normals and tangents
        print
        print('{:<30}'.format('Setting normals and tangents...'))
        self.get_element_normals_and_tangents()

        # link coefficients
        print
        print('{:<30}'.format('Setting link coefficients...'))
        self.get_link_coefficients()

        print
        print('{:<30}'.format('Setting boundary and interior lists...'))

        # boundary elements
        self.belems    = np.where(self.neighbors == -1)[0]
        interior_elems = []
        for i in range(self.nelements):
            if i not in self.belems:
                interior_elems.append(i)

        self.interior_elems = np.asarray(interior_elems)

        # boundary nodes
        bnodes = []
        interior_nodes = []
        for bface in self.bfaces:
            for node in self.xfaces[bface]:
                if node not in bnodes:
                    bnodes.append(node)

        for i in range(self.npoints):
            if i not in bnodes:
                interior_nodes.append(i)

        self.bnodes         = np.asarray(bnodes)
        self.interior_nodes = np.asarray(interior_nodes)

        print
        print('{:<30}'.format('Setting Dirichlet, Neumann, Robin, & Adiabatic Faces...'))

        # Dirichlet and Neumann faces
        dirichlet_faces = []; dirichlet_faces_val = []
        neumann_faces   = [];  robin_faces     = []
        adiabatic_faces = []
        for tag in self.tags:
            tagdim, tagval, physval = self.tags[tag]
            if tagdim == self.dim-1:
                if tag.startswith('Dirichlet'):
                    for i in range(self.nfaces):
                        if self.face_markers[i] == tagval:
                            dirichlet_faces.append(i)
                            dirichlet_faces_val.append(physval)

                if tag.startswith('Neumann'):
                    for i in range(self.nfaces):
                        if self.face_markers[i] == tagval:
                            neumann_faces.append(i)

                if tag.startswith('Robin'):
                    for i in range(self.nfaces):
                        if self.face_markers[i] == tagval:
                            robin_faces.append(i)
                                               
                if tag.startswith('Adiabatic'):
                    for i in range(self.nfaces):
                        if self.face_markers[i] == tagval:
                            adiabatic_faces.append(i)
                            

        self.dirichlet_faces     = np.asarray(dirichlet_faces)
        self.dirichlet_faces_val = np.asarray(dirichlet_faces_val)

        self.neumann_faces     = np.asarray(neumann_faces)

        self.robin_faces     = np.asarray(robin_faces)

        self.adiabatic_faces = np.asarray(adiabatic_faces)

        print
        print('{:<30}'.format('Mesh Pre-processing done...'))

    def get_bbox(self):
          x = self.points[:, 0]
          y = self.points[:, 1]
          z = self.points[:, 2]

          return  BBox(x.min(), x.max(), y.min(), y.max(), z.min(), z.max())

    def find_elem_type(self):
        f = open(self.mshfile, 'r')

        while(True):
            l = f.readline()
            if l.startswith('$Elements'):
                break

        nelements = int(f.readline())
        for i in range(nelements-1):
            l = f.readline()
        
        data = [ int(j) for j in f.readline().split()[1:] ]

        elem_type = data[0]
        
        elem_types = {2:'tri',3:'quad'}
        self.elem_type = self.etype = elem_types[data[0]]
        
        if self.elem_type == 'tri':
            self.dim          = 2
            self.elem_nfaces  = 3
            self.elem_nnodes  = 3
            self.face_nnodes  = 2

        if self.elem_type == 'quad':
            self.dim          = 2
            self.elem_nfaces  = 4
            self.elem_nnodes  = 4
            self.face_nnodes  = 2
        
        f.close()

    def read_nodes_and_elements(self):
        self.read_nodes()
        self.read_elements()

    def read_nodes(self):
        f = open(self.mshfile, 'r')
        f.seek(0)
        while True:
            l = f.readline()
            if l.startswith('$Nodes'):
                break
        
        self.nnodes = nnodes = int(f.readline())
        points = np.ones( shape=(nnodes, 3), dtype=float )

        for i in range(nnodes):
            points[i] = [ float(j) for j in f.readline().split()[1:1+3] ]
            
        self.points = np.array(points)[:,:]
        self.npoints = self.points.shape[0]

        self.x = self.points[:, 0]
        self.y = self.points[:, 1]
        self.z = self.points[:, 2]

        f.close()

    def read_elements(self):
        f = open(self.mshfile, 'r')
        f.seek(0)
        while True:
            l = f.readline()
            if l.startswith('$Elements'):
                break
        
        self.nelements_total = nelements_total = int(f.readline())

        element_data = np.ones( shape=(nelements_total, 25), dtype=int ) * -1
        mask = np.ones( shape=(nelements_total, 25), dtype=bool )

        self.elements_total = ma.masked_array(data=element_data, mask=mask)
        
        for i in range(nelements_total):
            data = [ int(j) for j in f.readline().split()[1:] ]

            self.elements_total.data[i][:len(data)] = data
            self.elements_total.mask[i][:len(data)] = False

            # convert from GMSH 1 based indexing to Python 0 based indexing
            ntags = data[1]
            self.elements_total.data[i][2+ntags:] -= 1
            self.elements_total.data[i][2+ntags+self.elem_nnodes:] -= 1022

        # read the different types of elements
        self.lines = lines = self.elements_total[np.where(self.elements_total[:,0] == 1)[0]]
        self.tri   = tri   = self.elements_total[np.where(self.elements_total[:,0] == 2)[0]]
        self.quads = quads = self.elements_total[np.where(self.elements_total[:,0] == 3)[0]]
        self.tets  = tets  = self.elements_total[np.where(self.elements_total[:,0] == 4)[0]]
        self.hexs  = hexs  = self.elements_total[np.where(self.elements_total[:,0] == 5)[0]]

        self.nlines = Nlines = len(lines)
        self.ntri   = Ntri   = len(tri)
        self.nquads = Nquads = len(quads)
        self.ntets  = Ntets  = len(tets)
        self.nhexs  = Nhexs  = len(hexs)

        # concatenate the elements into one data structure
        if self.dim == 2:
            if quads.size == 0:
                self.elements = self.tri
            elif tri.size == 0:
                self.elements = self.quads
            else:
                self.elements = np.concatenate( (self.tri, self.quads) )
        else:
            if hexs.size == 0:
                self.elements = self.tets
            elif tets.size == 0:
                self.elements = self.hexs
            else:
                self.elements = np.concatenate( (self.tets, self.hexs) )

        self.nelements = self.elements.shape[0]

        # sort element data
        xc = np.zeros(shape=(self.nelements))
        yc = np.zeros(shape=(self.nelements))
        zc = np.zeros(shape=(self.nelements))

        enodes = np.ones(shape=(self.nelements, self.elem_nnodes), dtype=int)
        for i in range(self.nelements):
            data = self.elements.data[i][np.where(self.elements.mask[i] == False)[0]]
            enodes[i] = self.elements.data[i][np.where(self.elements.mask[i] == False)[0]][-self.elem_nnodes:]

        # fix ordering for negative jacobian elements
        if self.etype == 'tri':
            for i in range(self.nelements):
                pts = self.points[ enodes[i] ]
                xv = pts[:, 0]; yv = pts[:, 1]; zv = pts[:, 2]
                
                jac = (xv[1]-xv[0])*(yv[2]-yv[0]) - (xv[2]-xv[0])*(yv[1]-yv[0])
                if jac < 0:
                    n1, n2, n3 = enodes[i]
                    enodes[i,0] = n2
                    enodes[i,1] = n1

        elif self.etype == 'quad':
            for i in range(self.nelements):
                pts = self.points[enodes[i]]
                xv = pts[:-1, 0]; yv = pts[:-1, 1]; zv = pts[:-1, 2]
                
                jac = (xv[1]-xv[0])*(yv[2]-yv[0]) - (xv[2]-xv[0])*(yv[1]-yv[0])
                if jac < 0:
                    enodes[i] = enodes[i][::-1]

        self.element2vertices = enodes
        element_sizes = np.zeros(shape=(self.nelements))

        # get the element centroids
        for i in range(self.nelements):
            cents = 1./self.elem_nnodes * sum(self.points[enodes[i]], 0)
            
            xc[i] = cents[0]; yc[i] = cents[1]
            if self.dim == 3:
                zc[i] = cents[2]

            _enodes   = self.points[enodes[i]]
            max_dist = -1
            for j in range(self.elem_nnodes):
                dist_to_node = np.sqrt( (xc[i]-_enodes[j, 0])**2 + \
                                        (yc[i]-_enodes[j, 1])**2 + \
                                        (zc[i]-_enodes[j, 2])**2 )

                max_dist = max(max_dist, dist_to_node)

            element_sizes[i] = max_dist

        self.xc = xc
        self.yc = yc
        self.zc = zc

        self.cents = np.empty(shape=(xc.size, 3), dtype=np.float64)
        self.cents[:, 0] = xc
        self.cents[:, 1] = yc
        self.cents[:, 2] = zc

        self.element_sizes = element_sizes
        f.close()

    def _read_physical_tags(self):
        f = open(self.mshfile, 'r')
        f.seek(0)
        
        while True:
            l = f.readline()
            if l.startswith('$PhysicalNames'):
                break
            if l == '':
                print ('Physical tags not found. At EOF, Exiting ...')
                sys.exit()

        n_physical_tags = int( f.readline() )

        tags = {}
        for i in range(n_physical_tags):
            l = f.readline().split(' ')
            tagdim, tagval, physval  = [int (j) for j in l[:3]]
            tagstr = l[3][1:-2].replace('"', '')
            tags[ tagstr ] = (tagdim, tagval, physval)
            
        print 'Physical Boundaries in the Mesh'
        dim = self.dim
        for key, val in tags.iteritems():
            if val[0] == dim-1:
                print("TagName = {:<10} Tagval = {:02d}".format(key, val[1]))

        self.tags = tags
        f.close()

    def get_bfaces(self, tag):
        tagdim = self.tags[tag][0]
        tagval = self.tags[tag][1]

        if tagdim == self.dim:
            raise ValueError("get_bfaces::tag dim same as problem dim")

        if self.dim == 2:

          msh_lines_data = self.lines.data
          lines = np.zeros( (self.nlines, 2), dtype=np.int )

          counter = -1
          for i in range(self.nlines):
              if msh_lines_data[i][2] == tagval:
                  counter += 1
                  lines[counter][0] = msh_lines_data[i][4]
                  lines[counter][1] = msh_lines_data[i][5]

        if self.dim == 3:
            if self.etype == 'hex':
                msh_quads_data = self.quads.data
                lines = np.zeros( (self.nquads, 4), dtype=np.int )

                counter = -1
                for i in range(self.nquads):
                    if ( msh_quads_data[i][2] == tagval ):
                        counter += 1
                        lines[counter][0] = msh_quads_data[i][4]
                        lines[counter][1] = msh_quads_data[i][5]
                        lines[counter][2] = msh_quads_data[i][6]
                        lines[counter][3] = msh_quads_data[i][7]

            elif self.etype == 'tet':
                msh_tri_data = self.tri.data
                lines = np.zeros( (self.ntri, 3), dtype=np.int )
                
                counter = -1
                for i in range(self.ntri):
                    if ( msh_tri_data[i][2] == tagval ):
                        counter += 1
                        lines[counter][0] = msh_tri_data[i][4]
                        lines[counter][1] = msh_tri_data[i][5]
                        lines[counter][2] = msh_tri_data[i][6]

        return lines[:counter+1]

    def setup_connectivity(self):

        # number of faces for the element and the number of nodes for
        # each face
        elem_nface  = self.elem_nfaces
        face_nnodes = self.face_nnodes

        nfaces  = 0
        xfaces = np.zeros( shape=(self.nelements_total*elem_nface, face_nnodes), dtype=np.int64 )

        f2e    = np.ones( shape=( self.nelements*elem_nface, 2), dtype=np.int64 ) * -1
        f2ef   = np.ones( shape=( self.nelements*elem_nface, 2), dtype=np.int64 ) * -1
        e2f    = np.ones( shape=( self.nelements, elem_nface), dtype=np.int64)    * -1

        neighbors = np.ones( shape=(self.nelements, elem_nface), dtype=np.int64 ) * -1

        faces_done = np.zeros( shape=(self.nelements,), dtype=np.int64 )

        # boundary faces
        nbfaces = 0
        bfaces = np.zeros(shape=(self.nelements*elem_nface), dtype=np.int64)

        rad_scales = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0]

        for elem_index in range(self.nelements):
            if (faces_done[elem_index] < elem_nface):

                for face_index in range(elem_nface):

                    # get the neighbor for this face for the current element
                    xface, nbr_index = self._get_face_neighbor(elem_index, face_index)

                    # if a face neighbor exists for this face, it must
                    # be an interior face. We find the reciprocal for
                    # the neighbor to set up the arrays
                    #if nbr_index.size > 0:
                    if nbr_index[0] != -1:
                        reciprocal_found = False

                        for rad_scale in rad_scales:

                            for i in range(elem_nface):
                                tmp_xface, tmp_elem_index = self._get_face_neighbor(nbr_index[0], i, rad_scale)

                                if (tmp_elem_index[0] != -1 and tmp_elem_index[0] == elem_index):
                                    nbr_face_index = i
                                    reciprocal_found = True

                                if reciprocal_found:
                                    break

                            if reciprocal_found:
                                break

                        if (not reciprocal_found):
                            raise RuntimeError("Somethings not right with the mesh")

                        # check if the face has been dealt with or not
                        if ( neighbors[elem_index][face_index]        == -1 and \
                             neighbors[nbr_index[0] ][nbr_face_index] == -1 ):

                            # this is a new face, so we add it to the list
                            f2e[nfaces][0] = elem_index
                            f2e[nfaces][1] = nbr_index[0]

                            f2ef[nfaces][0] = face_index
                            f2ef[nfaces][1] = nbr_face_index

                            e2f[elem_index][  face_index]     = nfaces
                            e2f[nbr_index[0] ][nbr_face_index] = nfaces

                            neighbors[elem_index][  face_index]    = nbr_index[0]
                            neighbors[nbr_index[0]][nbr_face_index] = elem_index

                            # add the x-faces array
                            for i in range(self.face_nnodes):
                                xfaces[nfaces][i] = xface[i]

                            faces_done[elem_index] += 1
                            faces_done[nbr_index ] += 1

                            nfaces += 1                        

                    else:
                        # this is a boundary face and needs to be added
                        f2e[nfaces][0] = elem_index
                        e2f[elem_index][face_index] = nfaces

                        for i in range(self.face_nnodes):
                            xfaces[nfaces][i] = xface[i]

                        f2ef[nfaces][0] = face_index

                        faces_done[elem_index] += 1

                        # add this index to the list of boundary faces
                        bfaces[nbfaces] = nfaces
                        nbfaces += 1

                        # increment the number of faces counter
                        nfaces += 1

        self.nfaces    = nfaces
        self.e2f       = e2f
        self.neighbors = neighbors
        
        self.f2e    = f2e[:nfaces]
        self.f2ef   = f2ef[:nfaces]
        self.xfaces = xfaces[:nfaces]

        self.nbfaces = nbfaces
        self.bfaces  = bfaces[:nbfaces]

        tag_map = {
            'ZeroGradient':  1000,
            'ReflectiveWall':1001,
            'Inlet':1002,
            'NoSlipAdiabaticWall':1003,
            'SubsonicOutlet':1004,
            'SubsonicInlet':1005,
            'NoSlipIsothermalWall':1006,
            'Characteristic':1007,
            'Left':1,
            'Right':2,
            'Bottom':3,
            'Top':4,
            'Back':5,
            'Front':6,
            'periodic_lo_x':2001,
            'periodic_hi_x':2002,
            'periodic_lo_y':2003,
            'periodic_hi_y':2004,
            'periodic_lo_z':2005,
            'periodic_hi_z':2006}

        # now go over the boundary faces and store the type in the face marker
        face_markers = np.ones( nfaces, int ) * -1
        bface_centroids = np.zeros( shape=(nbfaces, 3) )

        for i in range(nbfaces):
            face_index    = bfaces[i]
            face_centroid = np.sum(self.points[self.xfaces[face_index]], axis=0)/self.face_nnodes

            bface_centroids[i][:] = face_centroid
            
            for index, tag in enumerate(self.btags.keys()):
                lines = self.btags[tag]
                tree  = self.btree[tag]

                tagdata = self.tags[tag]
                tagdim  = tagdata[0]
                tagval  = tagdata[1]

                # query the kd-tree to find nearest msh faces
                nbr_lines = np.asarray(tree.query_ball_point(face_centroid, 0.01))

                if nbr_lines.size > 0:
                    found_face = False

                    for line_index in nbr_lines:
                        line     = lines[line_index]
                        centroid = np.sum(self.points[line], axis=0)/self.face_nnodes

                        diff2 = (face_centroid[0] - centroid[0])**2 + \
                                (face_centroid[1] - centroid[1])**2 + \
                                (face_centroid[2] - centroid[2])**2

                        if (diff2 < 1e-12):
                            _tagval = tagval
                            if tag in tag_map.keys():
                                _tagval = tag_map[tag]
                                
                            face_markers[face_index] = _tagval
                            found_face = True
                        
                    if (found_face):
                        break
                        
        # this tells us what each boundary face is
        self.face_markers = face_markers

        # create a kdTree for this processor's boundary faces. This will be
        # used later to check for fast finding of local matching faces
        self.bface_tree = None
        if self.nbfaces > 0:
            self.bface_tree = cKDTree(bface_centroids)

        # set up the local faces
        nlocal_faces = 0
        local_faces  = np.ones(shape=(nfaces), dtype=np.int64) * -1
        for face_index in range(self.nfaces):
            if (self.face_markers[face_index] == -1):
                local_faces[nlocal_faces] = face_index
                nlocal_faces += 1

        self.nlocal_faces = nlocal_faces
        self.local_faces  = local_faces

        ######################################################################
        # SET UP FACE NORMALS FOR THE MESH
        normals     = np.zeros(shape=(nfaces, 3))
        normal_mags = np.zeros(shape=(nfaces))

        for face_index in range(nfaces):
            if self.etype in ['tri', 'quad']:
                xa, xb = self.points[self.xfaces[face_index]]
                
                delx = xb[0] - xa[0]
                dely = xb[1] - xa[1]
        
                mag = 1./np.sqrt( delx**2 + dely**2 )
                
                # store the unit normals and the magnitude
                delx *= mag
                dely *= mag
                
                normals[face_index][0] =  dely
                normals[face_index][1] = -delx
                
                normal_mags[face_index] = 1./mag

            elif self.etype == 'hex':
                xa, xb, xc, xd = self.points[self.xfaces[face_index]]
                
                A = xb - xa
                B = xc - xa
                
                nx = +(A[1]*B[2] - A[2]*B[1])
                ny = -(A[0]*B[2] - A[2]*B[0])
                nz = +(A[0]*B[1] - A[1]*B[0])

                mag = 1./np.sqrt(nx**2 + ny**2 + nz**2)

                normals[face_index][0] = nx*mag
                normals[face_index][1] = ny*mag
                normals[face_index][2] = nz*mag

                normal_mags[face_index] = 1./mag

        self.normals     = normals
        self.normal_mags = normal_mags

    def get_vertex_connectivity(self):
        e2v    = self.e2v
        nverts = self.npoints

        v2e  = np.ones((nverts, 50), dtype=np.int64) * -1
        Nv2e = np.zeros(nverts, dtype=np.int64)

        idw_e2v = np.zeros((nverts, 50))

        for i in range(nverts):
            elems = np.where( e2v == i )[0]

            Nv2e[i]             = elems.size
            v2e[i, :elems.size] = elems

            xv = self.points[i]

            di_sum = 0.0
            for j in range(elems.size):
                elem_index = elems[j]
                
                xc = self.cents[elem_index]

                di = 1.0/np.linalg.norm(xc-xv)
                
                idw_e2v[i, j] = di
                di_sum += di

            # normalize
            idw_e2v[i,:] /= di_sum

        # store the arrays
        self.Nv2e    = Nv2e
        self.v2e     = v2e
        self.idw_e2v = idw_e2v

    def get_e2f_idw_coefficients(self):
        nfaces = self.nfaces
        
        # face centroids
        self.face_centroids = fc = np.sum( self.points[self.xfaces], axis=1 )/self.dim

        bfaces = self.bfaces
        faces  = self.faces = np.zeros(self.nfaces-bfaces.size, np.int64)

        idw_e2f = np.zeros(shape=(nfaces,2))

        counter = -1
        for i in range(nfaces):
            if i not in bfaces:
                xf = fc[i]

                left, rght = self.f2e[i]            

                xc_left = self.cents[left]
                xc_rght = self.cents[rght]
                
                di_left = 1.0/np.linalg.norm( xf-xc_left )
                di_rght = 1.0/np.linalg.norm( xf-xc_rght )
                
                idw_e2f[i, 0] = di_left/(di_left + di_rght)
                idw_e2f[i, 1] = 1.0 - idw_e2f[i, 0]

                counter += 1
                faces[counter] = i

        self.idw_e2f = idw_e2f

    def get_element_volumes(self):
        e2v     = self.e2v
        volumes = np.zeros(e2v.shape[0])

        if self.etype == 'tri':
            for i in range(self.nelements):
                pts = self.points[ e2v[i] ]
                xv = pts[:, 0]; yv = pts[:, 1]; zv = pts[:, 2]
                
                jac = (xv[1]-xv[0])*(yv[2]-yv[0]) - (xv[2]-xv[0])*(yv[1]-yv[0])
                
                volumes[i] = jac/2.0

        self.volumes = volumes
        self._check_element_volumes()

    def get_element_normals_and_tangents(self):
        self.element_normals  = element_normals  = np.zeros(shape=(self.nelements, self.elem_nfaces, 3))
        self.element_tangents = element_tangents = np.zeros(shape=(self.nelements, self.elem_nfaces, 3))

        for i in range(self.nelements):
            faces = self.e2f[i]
            for j, face in enumerate(faces):
                
                xa, xb = self.points[self.xfaces[face]]
                ds = self.face_areas[face]

                element_tangents[i,j,:] = (xb-xa)/ds

                if self.f2e[face, 0] == i:
                    element_normals[i, j, :] = self.normals[face]
                else:
                    element_normals[i, j, :] = -self.normals[face]

    def get_link_coefficients(self):
        self.elinks = elinks = np.zeros(shape=(self.nelements, self.elem_nfaces, 2))

        for i in range(self.nelements):
            nbrs = self.neighbors[i]

            xc = self.cents[i]

            for j, nbr in enumerate(nbrs):
                if (nbr > -1):
                    xc_nbr = self.cents[nbr]
                else:
                    face = self.e2f[i, j]
                    xc_nbr = self.face_centroids[face]

                vec_diff = xc_nbr - xc

                normal  = self.element_normals[ i, j]
                tangent = self.element_tangents[i, j]

                elinks[i, j, 0] = normal.dot( vec_diff)
                elinks[i, j, 1] = tangent.dot(vec_diff)

    def _check_element_volumes(self):
        vol = self.vol = np.zeros(self.nelements)
        
        e2f        = self.e2f

        face_cents = self.face_centroids
        face_areas = self.face_areas = np.zeros(shape=(self.nfaces))
        
        for elem_index in range(self.nelements):
            faces = self.e2f[elem_index]

            sx = 0.0
            sy = 0.0

            for face in faces:
                if self.f2e[face, 0] == elem_index:
                    normal = self.normals[face]
                else:
                    normal = -1*self.normals[face]
                    
                fc = face_cents[face]
                
                xa, xb = self.points[self.xfaces[face]]
                ds = np.linalg.norm(xa-xb)

                sx += ds*fc[0]*normal[0]
                sy += ds*fc[1]*normal[1]
                
                face_areas[face] = ds

            vol[elem_index] = 0.5 * (sx + sy)                

    def _get_face_neighbor(self, elem_index, face_index, rad_scale=5.0):
        vertices = self.element2vertices[elem_index]
        faces = utils.round_trip_connect( vertices, self.dim, self.etype )

        face = faces[face_index]

        ret_val = np.array([-1])

        tree      = self.kd_tree
        query_pnt = np.zeros(3)
        
        for i in range(self.face_nnodes):
            query_pnt += self.points[ face[i] ]
        
        # face centroid used for neighbor search
        query_pnt *= 1./self.face_nnodes
        
        element_centroid = self.cents[elem_index]

        #query_rad  = self.element_sizes[elem_index] * 1.5
        query_rad = np.sqrt( (query_pnt[0]-element_centroid[0])**2 + \
                             (query_pnt[1]-element_centroid[1])**2 + \
                             (query_pnt[2]-element_centroid[2])**2 )

        nbrs = np.asarray( tree.query_ball_point(query_pnt, query_rad*rad_scale) )
        self_indices = np.where(nbrs == elem_index)
        nbrs = np.delete(nbrs, self_indices)

        ret_val[0] = utils.get_face_neighbor(
            self.dim, self.etype,
            self.elem_nnodes, self.elem_nfaces, self.face_nnodes,
            nbrs.size, nbrs, query_pnt,
            self.element2vertices, self.points)

        return face, ret_val

    ###################################################################
    # Helper functions for plotting in 2D
    def plot_all_elems(self, color='b'):
        for i in range(self.nelements):
            self.plot_elem(i, color)

    def plot_elem(self, elem_index, color):
        points = self.points[self.element2vertices[elem_index]]
        
        x = points[:,0]; y = points[:, 1]
        plt.fill(x,y, color)

    def plot_face(self, face_index):
        points = self.points[self.xfaces[face_index]]
        x = points[:,0]; y = points[:, 1]
        plt.plot(x, y, c='r', linewidth=5)

    def plot_node(self, node_index):
        points = self.points[node_index]

def write_vtk_ugrid(fname, msh, phi):
    import vtk
    
    if msh.etype == 'tri':
        vtk_elem_type = vtk.VTK_TRIANGLE
    elif msh.etype == 'hex':
        vtk_elem_type = vtk.VTK_HEXAHEDRON
    elif msh.etype == 'tet':
        vtk_elem_type = vtk.VTK_TETRA
    elif msh.etype == 'quad':
        vtk_elem_type = vtk.VTK_QUAD

    vtk_points = vtk.vtkPoints()
    for i in range(msh.npoints):
        _pts = msh.points[i]
        vtk_points.InsertNextPoint(_pts[0], _pts[1], _pts[2])

    vtk_cells = vtk.vtkCellArray()
    for i in range(msh.nelements):
        if msh.etype == 'tri': elem = vtk.vtkTriangle()
        if msh.etype == 'hex': elem = vtk.vtkHexahedron()
        if msh.etype == 'tet': elem = vtk.vtkTetra()
        if msh.etype == 'quad': elem = vtk.vtkQuad()
            
        e2v = msh.element2vertices[i]
        nverts = e2v.size
        
        point_ids = elem.GetPointIds()
        
        for j in range(nverts):
            point_ids.SetId(j, e2v[j])
            
        vtk_cells.InsertNextCell(elem)

    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(vtk_points)
    ug.SetCells(vtk_elem_type, vtk_cells)

    # rank data
    vtk_array = vtk.vtkFloatArray()
    vtk_array.SetNumberOfComponents(1)
    vtk_array.SetNumberOfTuples(ug.GetNumberOfCells())
    for i in range(ug.GetNumberOfCells()):
        vtk_array.SetValue(i, phi[i])

    vtk_array.SetName("phi")
    ug.GetCellData().AddArray(vtk_array)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetDataModeToBinary()

    if not fname.endswith('.vtu'): fname = fname + '.vtu'
    writer.SetFileName(fname)

    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(ug)
    else:
        writer.SetInputData(ug)
    
    writer.Write()
