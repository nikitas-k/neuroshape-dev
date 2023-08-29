import numpy as np

from mesh import mesh_io as io
from mesh import mesh_operations as mo
from lapy import Solver, TriaMesh, TetMesh
from mesh import attributes as ma
from utils._imports import import_optional_dependency

class Shape:
    """Core class of meshes for computation of eigenmodes and reconstructions. 
    Internal functions and attributes use support functions contained elsewhere.
    
    Efficient representation of mesh data structure with core
    functionality using sparse matrices internally (Scipy). Based on TriaMesh
    and TetMesh in lapy. Note: many of these operations are one-way and a copy
    of the original mesh is not made. If you wish to preserve the original surface, 
    write the surface out to a file before doing anything irreversible or 
    make a deepcopy of the class object.
    
    Can also track history of commands run from or to the class object - see 
    `Shape._history`
     
    @author Nikitas C. Koussis
    
    Many methods and routines are derived from Brainspace and lapy. Please
    be sure to cite their work as [S1], [S2], [S3], and [S4] if you use this
    class object.
     
    Parameters
    ----------
    polydata : array_like, file_like
        List of lists of 3 float coordinates and list of lists of 3 (for
        triangular mesh) or 4 (for tetrahedral mesh) int of indices, in 
        FreeSurfer triangular surface format (pial, thickness, white, etc.), 
        .gii format (GIFTI), gmsh object, TriaMesh, TetMesh, PLY file,
        Brainspace BSPolyData object, VTK file. See `neuroshape.io.iosupport`
    mask : array_like or file_like (optional)
        Indices to remove from `v`
    
    Attributes
    ----------
    v : array_like
        List of lists of 3 float coordinates
    f : array_like
        List of lists of 3 int of indices (for triangular mesh), or 4 int of 
        indices (for tetrahedral mesh)
    e : csc_matrix
        Symmetric adjacency matrix as csc sparse matrix
    data : ndarray
        scalar value on each vertex
    mask : NoneType or ndarray of shape=(num_verts,)
        List of indices to remove from `v`
    num_verts : int
        Number of vertices in mesh
    num_faces : int
        Number of faces in mesh
    polyshape : int
        Shape of mesh, 3 if triangular, 4 if tetrahedral
    closed : bool
        Whether mesh is closed or open
    adj_sym : csc_matrix
        Symmetric adjacency matrix as csc sparse matrix, equivalent to `e`
    
    In/out
    ------
    read_surface : 
        Binds VTK reader submethods to read surface from filename
    write_surface : 
        Binds VTK writer submethods to write surface to filename
    read_data : array_like or file_like
        Reads data from array or text file and maps to `data`
    write_data :
        Writes data from `data` to filename
    read_mask : array_like or file_like
        Indices to mask from `f` and `v` - creates ShapeCut as 
        inheritor subclass
    write_mask : 
        Binds VTK writer submethods to write masked surface to filename
    read_emodes : array_like or file_like
        Precomputed eigenmodes from file or ndarray
    write_emodes : 
        Write out emodes to filename
    read_evals : array_like or file_like
        Precomputed eigenvalues from file or ndarray
    write_evals :
        Write out evals to filename
        
    Mesh attribute functions
    ------------------------
    get_verts :
        Return the vertices in mesh as ndarray, shape=(num_verts,3)
    get_faces : 
        Return the faces in mesh as ndarray, shape=(num_faces,3)
    get_edges :
        Return the vertices and adjacent triangle ids for each edge
    get_polyshape :
        return whether the mesh is "triangular" (`t` shape=(num_faces,3)) or
        "tetrahedral" (`t` shape=(num_faces,4))
    get_closed :
        return whether the mesh is closed "True" or open "False"
    construct_adj_sym :
        construct symmetric edge adjacency matrix
    
    Vertex operations
    -----------------
    sort_verts :
        Returns sorted vertices
    find_vertex_correspondence :
        For each vertex find the corresponding vertex in given surface
    vertex_neighbors :
        Returns the neighboring vertices of a given vertex
    has_free_vertices :
        Check if the vertex list in a tetrahedral mesh has more vertices than
        it should    
    remove_free_vertices :
        Remove free vertices from the mesh
    vertex_degrees : staticmethod
        Compute the vertex degrees (number of edges at each vertex)
    vertex_areas : staticmethod
        Compute the area associated to each vertex 
        (1/3 of one-ring trias or 1/4 of one-ring tetras)
    vertex_normals : staticmethod
        Compute the vertex normals    
    geodesic_distance : 
        Find the geodesic distance of a vertex to another vertex
    geodesic_knn : 
        Find the geodesic distances of a vertex to its k-nearest
        neighbors
    
    Face operations
    ---------------
    not implemented
    
    Mesh operations
    ---------------    
    apply_mask :
        Apply mask to surface and make cuts, sets inheritor object ShapeCut
        with new surface
    threshold_surface :
        Selection of points or faces by 'distance', 'variance', or `kernel_type`
    threshold_minimum :
        Threshold vertices and faces by minimum value needed to remain connected
    combine_surfaces :
        Combine surfaces given sequence of polydata or Shape class
    split_surface :
        Split surface according to labelling
    downsample :
        Downsample surface according to labelling
    normalize : 
        Normalize mesh by `norm_types`
    unit_normalize : 
        Normalize mesh to unit surface area and centroid at origin
    refine : 
        Refine the mesh by placing new vertex on each edge midpoint
    orient : 
        Re-orient mesh to be consistent
    smooth : 
        Smooth the mesh iteratively in-place
    decimate :
        Decimation of vertices and faces according to decimation factor
    decimate_pro :
        Decimation of vertices and faces according to decimation factor,
        uses pyvista decimate_pro method
        
    Mesh/graph properties and metrics
    ---------------------------------
    get_connected_components :
        Get connected components
    isoriented :
        Check if tetrahedral mesh is oriented
    average_edge_length : staticmethod
        Get average edge lengths
    area : staticmethod
        Compute the total surface area of the mesh
    volume : staticmethod
        Compute the volume of closed triangular mesh or tetrahedral mesh
    boundary_loops : staticmethod
        Compute a tuple of boundary loops
    centroid : staticmethod
        Compute centroid of mesh
    curvature : staticmethod
        Compute curvature at vertices
    shortest_path : 
        Find the shortest path length of the mesh
    modularity : 
        Return the Newman spectral modularity coefficient of the mesh
    clustering_coefficient :
        Return the clustering coefficient of the mesh
    rich_club :
        Return the rich-club coefficient of the mesh at a particular degree
    geodesic_distmat :
        Return the geodesic distance matrix of the mesh using a heat
        diffusion kernel
    euclidean_distmat :
        Return the euclidean distance matrix of the mesh
        
    Mesh data operations
    --------------------
    parcellate_data :
        Parcellate vertex data based on a given parcellation
    threshold_data :
        Threshold vertex data based on `threshold_type` and `threshold`
    data_to_mask :
        Assign data to mask
    optimize_recon :
        Reconstruct data using optimal combinations of `emodes` iteratively
        from 0 to total modes, returning pearson correlation of reconstruction 
        to original map in `data`. Standardizes data and reconstruction to
        compute `pearsonr`. Based on reconstruction process in [1]
    reconstruct_data :
        Reconstruct `data` with `emodes`, minimizing ordinary least-squares 
        (OLS) error
     
    Triangular mesh specific operations
    -----------------------------------
    to_tetrahedral :
        Get tetrahedral mesh inside bounded mesh
    tria_normals :
        Compute triangle normals
    tria_qualities :
        Compute triangle quality for each triangle
    tria_areas :
        Compute the area of triangles using Heron's formula
    euler :
        Compute Euler characteristic
    is_manifold :
        Check if triangle mesh is manifold (no edges with >2 triangles)
    
    Tetrahedral mesh specific operations
    ------------------------------------
    to_triangular :
        Get boundary triangle mesh of tetrahedra
    
    ShapeDNA attributes and operations
    ----------------------------------
    num_modes : int, default 200
        number of modes to compute on the surface
    evals : ndarray, shape = (num_modes,)
        Eigenvalues computed on the surface
    emodes : ndarray, shape = (num_verts, num_modes)
        Eigenmodes computed on the surface - if `mask`, only computes eigenmodes
        on unmasked indices of `v` and `t`
    compute_eigenmodes :
        Compute eigenmodes `emodes` and eigenvalues `evals` given `num_modes`
        after surface passes checks
    Mesh : TriaMesh or TetMesh
        Inherited class from lapy, depends on `self.polyshape`
    FEM : Solver
        lapy Solver class computed on mesh, returns Finite Element Method.
        
    References
    ----------
    [S1]
    
    [S2]
    
    [S3]
    
    [S4]
    
    """
    
    def __init__(self, vertices, faces, data=None, mask=None, num_modes=200):
        # read polydata
        self.v, self.f = vertices, faces
        
        self.polyshape = self.get_polyshape()
        self.num_verts = np.max(self.v.shape)
        self.num_faces = np.max(self.f.shape)
        
        self.e = self._construct_adj_sym()
        self.adj_sym = self.e
        self.data = data
        
        if mask:
            self.ShapeCut = self.apply_mask(self)
        
        self.closed = self.get_closed()
        
        # checks
        if np.max(self.f) >= self.num_verts:
            raise ValueError("Max index of faces exceeds number of vertices")
        #if self.polyshape != 3 or self.polyshape != 4:
            #raise ValueError("Faces should have 3 (triangular) or 4 indices (tetrahedral)")
        if self.v.shape[1] != 3:
            raise ValueError("Vertices should have 3 coordinates")
            
        if self.polyshape == 3:
            self.Mesh = TriaMesh(self.v, self.f)
        else:
            self.Mesh = TetMesh(self.v, self.f)
        
        try:
            import_optional_dependency("sksparse.cholmod.cholesky")
            self.use_cholmod=True
        except:
            self.use_cholmod=False
            print("Scikit-sparse libraries not found, using LU decomposition for eigenmodes (slower)")
            
            
        self.FEM = Solver(self.Mesh, lump=True, use_cholmod=self.use_cholmod)
        self.num_modes = num_modes
        
        self.adj_sym = self._construct_adj_sym()
        
    ##############
    ##### IO #####
    ##############
    
    @classmethod
    def read_surface(cls, polydata):
        """
        Reads surface given by array_like or file_like `obj`, takes input
        filename with extension '.gii', '.vtp', '.vtk', '.ply', '.asc'.

        Parameters
        ----------
        polydata : str or tuples of ndarray of (num_verts, 3), (num_faces, 3)
            Input surface.

        Returns
        -------
        (num_verts, 3) ndarray
            Vertices of input mesh from `filename`
        (num_faces, 3) ndarray or (num_faces, 4) ndarray
            Faces of input mesh from `filename`

        """
        return Shape(*io.read_surface(polydata))
    
    def write_surface(self, filename):
        """
        Writes surface using writer, guesses format given extension 
        in `filename`.

        Parameters
        ----------
        filename : str
            Output filename

        """
        io.write_surface(self, filename)
        
    def read_data(self, obj):
        """
        Reads array_like or file_like data given by `obj`. Overwrites existing
        data.

        Parameters
        ----------
        obj : array_like or file_like
            Data to read in and save as vertex weights

        Returns
        -------
        ndarray of shape=(num_verts,)
            Vertex weights

        """
        data = io.read_data(obj)
        if data.shape[-1] != self.num_verts:
            # try transpose
            if data.T.shape[-1] != self.num_verts:
                raise ValueError('Data array cannot have more points than vertices')
            data = data.T
        
        self.data = data
    
    def write_data(self, filename):
        """
        Writes array_like `data` to file given by `filename`

        Parameters
        ----------
        filename : str
            Output file for vertex weights. File format guessed by filename
            extension, otherwise saved as .txt

        """
        io.write_data(self.data, filename)
        
    def read_mask(self, obj):
        """
        Reads mask from array_like or file_like. List of ints of indices
        `v` to mask from surface. Overwrites existing mask.

        Parameters
        ----------
        obj : array_like or file_like
            DESCRIPTION.

        Returns
        -------
        ndarray of type bool and shape=(m,)

        """
        self.mask = io.read_mask(obj)
    
    
    def write_mask(self, filename):
        """
        Binds writer submethods to write masked surface to filename. Output
        format is guessed from extension.

        Parameters
        ----------
        filename : str
            Output filename.

        """
        io.write_surface(self.mask, filename)
    
    def read_emodes(self, obj):
        """
        Reads input eigenmodes and stores in class field `emodes`. Overwrites
        existing eigenmodes.

        Parameters
        ----------
        obj : ndarray or str
            Array or filename of precomputed eigenmodes. Must have the same
            number of rows as the surface, where the columns are the modes.

        """
        
        emodes = io.read_data(obj)
        if self.evals is None:
            print('Eigenmodes array loaded into surface, expecting matching eigenvalues')
        else:
            if emodes.shape[-1] != self.evals.shape[0]:
                raise AttributeError('Number of eigenmodes must be the same as number of eigenvalues')
            else:
                self.emodes = emodes
    
    def write_emodes(self, filename):
        """
        Write eigenmodes out to filename.

        Parameters
        ----------
        filename : str
            Filename to write eigenmodes out to.

        """
        io.write_data(self.emodes, filename)
        
    def read_evals(self, obj):
        """
        Reads input eigenvalues and stores in class field `evals`. Overwrites
        existing eigenvalues.

        Parameters
        ----------
        obj : array_like or file_like
            Eigenvalues to be imported.

        Returns
        -------
        ndarray of shape=(num_modes,)
            Eigenvalues as Shape.evals

        """
        
        evals = io.read_data(obj)
        if self.emodes is None:
            print('Eigenmodes array loaded into surface, expecting matching eigenvalues')
        else:
            if evals.shape[0] != self.emodes.shape[-1]:
                raise AttributeError('Number of eigenmodes must be the same as number of eigenvalues')
            else:
                self.evals = evals
    
    ############################
    # Mesh attribute functions #
    ############################
    
    def get_verts(self):
        """
        Return the vertices in mesh as ndarray, shape=(num_verts,3)
        
        Returns
        -------
        ndarray of shape=(num_verts, 3)
            Vertex array
        """
        return self.v
    
    def get_faces(self):
        """
        Return the faces in mesh as ndarray, shape=(num_faces,3)
        
        Returns
        -------
        ndarray of shape=(num_faces, 3)
            Faces array
        """
        return self.f
    
    def get_edges(self):
        """
        Return the adjacency edge graph.
        
        Returns
        -------
        adj_sym : csc_matrix
            Adjacency matrix of vertices and edges
        """
        return self.adj_sym
    
    def get_polyshape(self):
        """
        Return whether the mesh is "triangular" (`t` shape=(num_faces,3)) or
        "tetrahedral" (`t` shape=(num_faces,4))
        
        Returns
        -------
        int
            shape of last axis in `self.f`
            
        """
        return ma.get_polyshape(self)
    
    def get_closed(self):
        """
        Tests whether surface is closed (no boundary triangles or tetras)

        Returns
        -------
        bool

        """
        return ma.get_closed(self)
    
    #####################
    # Vertex operations #
    #####################
    
    def sort_verts(self, criterion='x'):
        """
        Returns sorted vertices
        
        Parameters
        ----------
        criterion : str
            The sorting criterion. Can be 'x', 'y', 'z', 'distance', and 
            (if self.data) 'data_weight'.
        
        Returns
        -------
        sorted_verts : np.ndarray
            The sorted vertex coordinates
            
        """
        return mo.sort_verts(self, criterion)
    
    def find_vertex_correspondence(self, ref_surf, eps=0, n_jobs=1):
        """
        For each point in the input surface find its corresponding point
        in the reference surface.

        Parameters
        ----------
        ref_surf : Shape
            Reference surface.
        eps : non-negative float, optional
            Correspondence tolerance. If ``eps=0``, find exact
            correspondences. Default is 0.
        n_jobs : int, optional
            Number of parallel jobs. Default is 1.

        Returns
        -------
        correspondence : ndarray, shape (n_points,)
            Array of correspondences (indices) with `num_verts` elements,
            where `num_verts` is the number of vertices of the input
            surface `surf`. Each entry indexes its corresponding
            point in the reference surface `ref_surf`.

        """
        return mo.find_vertex_correspondence(self, ref_surf, eps, n_jobs)
    
    def get_vertex_neighbors(self, vertex):
        """
        Returns the neighboring vertices of a given vertex.

        Parameters
        ----------
        vertex_index : int
            Index of the vertex.

        Returns
        -------
        list
            List of neighboring vertex indices.
        """
        return mo.vertex_neighbors(self, vertex)
    
    def has_free_vertices(self):
        """Check if the vertex list has more vertices than what is used.

        (same implementation as in `~lapy.TriaMesh`)
        
        Returns
        -------
        bool
            Whether vertex list has more vertices than what is used.
        """
        return mo.has_free_vertices(self)
    
    def remove_free_vertices(self):
        """Remove unused (free) vertices from v and t.

        These are vertices that are not used in any triangle. They can produce problems
        when constructing, e.g., Laplace matrices.

        Will update v and t in mesh.
        Same implementation as in `~lapy.TriaMesh`.

        Returns
        -------
        np.ndarray
            Indices (from original list) of kept vertices.
        np.ndarray
            Indices of deleted (unused) vertices.
        """
        return mo.remove_free_vertices(self)
    
    def vertex_degrees(self):
        """
        Compute the vertex degrees (number of edges at each vertex).

        Returns
        -------
        np.ndarray
            Array of vertex degrees.
        """
        return mo.vertex_degrees(self)
    
    def vertex_areas(self):
        """
        Compute the area associated to each vertex 
        (1/3 of one-ring trias or 1/4 of one-ring tetras)
            
        Returns
        -------
        np.ndarray
            Array of vertex areas.
        """
        return mo.vertex_areas(self)
    
    def vertex_normals(self):
        """
        Compute the vertex normals.
        Normals around each vertex are averaged, weighted
        by the angle that they contribute.
        Ordering is important: counterclockwise when looking
        at the triangle from above.
            
        Returns
        -------
        np.ndarray of shape (n_triangles, 3) or (n_tetrahedra, 4)
            Vertex normals.
        """
        return mo.vertex_normals(self)
    
    def shortest_path_length(self, source, target):
        """
        Calculates the shortest path length between two vertices in the mesh.

        Parameters
        ----------
        source : int
            Source vertex
        target : int
            Target vertex

        Returns
        -------
        float
            The shortest path length between source and target nodes. Returns np.inf
            if no path exists.
        
        """
        return mo.shortest_path_length(self, source, target)
    
    def geodesic_distance(self, source, target):
        """
        Find the geodesic distance of a vertex to another vertex, uses heat diffusion
        method. Based on [MO2].
            
        Parameters
        ----------
        source : int
            Source vertex
        target : int
            Target vertex

        Returns
        -------
        float 
            The geodesic distance between source and target nodes. Returns np.inf
            if no path exists.
        
        Notes
        -----
        Much faster than `shortest_path_length()` but cannot be computed on tetrahedra
        meshes.
        """
        return mo.geodesic_distance(self, source, target)
    
    def geodesic_knn(self, source, knn=100):
        """
        Find the geodesic distances of a vertex to its k-nearest
        neighbors using backward Euler solution in [MO2].
        
        Parameters
        ----------
        surf : Shape
            Mesh object
        source : int
            Source vertex
        knn : int
            Number of nearest neighbors to compute distances
            
        Returns
        -------
        ndarray of shape=(knn,)
            geodesic distances from the source vertex to its k-nearest neighbors
        """
        return mo.geodesic_knn(self, source, knn)        
    
    ###################
    # Face operations #
    ###################
    
    # none implemented
    
    ###################
    # Mesh operations #
    ###################
    
    def apply_mask(self, mask):
        """
        Mask surface vertices.

        Faces corresponding to these points are also kept.

        Parameters
        ----------
        mask : 1D ndarray or file-like
            Binary boolean array. Zero elements are discarded.

        Returns
        -------
        Shape
            Mesh object after masking

        """
        return Shape(*mo.apply_mask(self, mask))
    
    def parcellate_data(self, labels):
        """
        Parcellate vertex data based on a given parcellation.

        Parameters
        ----------
        labels : np.ndarray of shape=(N,).
            label array of ints

        Returns
        -------
        ndarray
            Parcellated data of shape (num_parcels, M). Also saves as attribute
            `self.parcellated_data`
        
        Notes
        -----
        `labels` must be integer label array or file-like with integer labels for
            each vertex.
            
        """
        self.parcellated_data = mo.parcellate_data(self, labels)
        
        return self.parcellated_data
    
    def threshold_surface(self, threshold_type='distance', threshold=0.1):
        """
        Selection of edges in mesh adjacency matrix based on `threshold_type`
        and `threshold`.

        Parameters
        ----------
        threshold_type : str, optional
            Thresholding type. The default is 'distance'.
        threshold : float, optional
            Weight to threshold based on `mesh.mesh_operations.threshold_type`. The default is 0.1.

        Raises
        ------
        ValueError
            `threshold_type` not in `mesh.mesh_operations.threshold_types`

        Returns
        -------
        csc_matrix
            Thresholded adjacency matrix
            
        Notes
        -----
        See `mesh.mesh_operations.threshold_types`

        """
        return mo.threshold_surface(self, threshold_type, threshold)
    
    def threshold_minimum(self, return_threshold=False):
        """
        Threshold surface adjacency matrix by minimum value needed to 
        remain connected. Based on [MO1].

        Parameters
        ----------
        return_threshold : bool
            Returns minimum threshold if "True"

        Returns
        -------
        Shape
            Thresholded surface
        float, optional
            
        
        References
        ----------
        #TODO [MO1] Cohen 2013

        """
        return mo.threshold_minimum(self)
            
        
    def combine_surfaces(self, *new_surfaces):
        """
        Combine surfaces given sequence of polydata or Shape class. Accepts
        file_likes as new surfaces to combine.
        
        Parameters
        ----------
        *new_surfaces : list
            Meshes to combine with original surface. Accepts str of file_like, 
            np.ndarray, or list of lists of vertices and faces.
        
        Returns
        -------
        Shape
            Combined surfaces
        """
        return Shape(mo.combine_surfaces(self, *new_surfaces))
    
    def split_surface(self, labels):
        """
        Split surface into separate meshes based on a label array.
        
        Parameters
        ----------
        labels : ndarray or str of file_like
            Label array of ints specifying the label for each vertex.
            
        Returns
        -------
        list
            List where each entry is a separate Shape object split according
            to the labels
        """
        return mo.split_surface(self, *labels)
        
    def normalize(self, norm_type='area'):
        """
        Normalize mesh by `norm_types`
        """
        return mo.normalize(self, norm_type)
        
    def unit_normalize(self):
        """
        Normalize mesh to unit surface area and centroid at origin
        """
        return mo.unit_normalize(self)
        
    def refine(self):
        """
        Refine the mesh by placing new vertex on each edge midpoint
        """
        return Shape(mo.refine(self))
        
    def orient(self):
        """
        Re-orient mesh to be consistent
        """
        return mo.orient(self)
        
    def smooth(self, fwhm=6.0):
        """
        Smooth the mesh iteratively in-place
        """
        return mo.smooth(self, fwhm)
        
    def remove_free_vertices(self):
        """
        Remove free vertices from the mesh
        """
        if not self.has_free_vertices():
            raise RuntimeWarning('Mesh has no free vertices')
            return
        
        return mo.remove_free_vertices(self)
        
    def decimate(self, decimation_factor=0.1):
        """
        Decimation of vertices and faces according to decimation factor,
        uses pyvista decimate method
        """
        return Shape(mo.decimate(self, decimation_factor))
    
    def decimate_pro(self, decimation_factor=0.1):
        """
        Decimation of vertices and faces according to decimation factor,
        uses pyvista decimate_pro method
        """
        return Shape(mo.decimate_pro(self, decimation_factor))
    
    def optimize_recon(self, direction='up', short=True):
        """
        Reconstruct data using optimal combinations of eigenmodes iteratively
        from 0 to total modes, returning correlation of reconstruction to original
        map in `data`. Based on reconstruction process in [1].

        Parameters
        ----------
        emodes : ndarray of shape=(n_vertices, n_modes)
            Eigenmodes to reconstruct data.
        data : ndarray of shape=(n_vertices,)
            Data to reconstruct.
        direction : str, optional
            Direction to perform reconstruction. 'up' adds modes, and reproduces
            the method used in Figure 1 of [1]. 'down' removes modes, reproducing
            the method in Figure 3 of [1]. The default is 'up'.
        short : bool, optional
            Whether to perform reconstruction first with shortest-wavelength modes
            or with longest-wavelength modes in `emodes`.
            The default is True.

        Returns
        -------
        reconstructed_corr : ndarray of shape=(n_modes,)
            Correlation metric (pearson) of reconstructed data at each mode
            in the processes described above.
            
        References
        ----------
        [1] 

        """
        
        return mo.optimize_recon(self, direction, short)
        
    def reconstruct_data(self, return_betas=False):
        """
        Reconstruct a dataset of `n_vertices` given a set of eigenmodes and coeffs
        conditioned on data using ordinary least squares (OLS)

        Parameters
        ----------
        surf : np.ndarray of shape (M,)
            Coefficients output from fitting OLS
        eigenmodes : np.ndarray of shape (n_vertices, M)
            Eigenmodes of `n_vertices` by number of eigenvalues M
            
        Returns
        -------
        new_data : np.ndarray of (n_vertices,)
            Reconstructed data

        """
        
        return mo.reconstruct_data(self, return_betas)

    def to_tetrahedral(self):
        """
        Get tetrahedral mesh inside bounded mesh
        """
        if self.polyshape == 4:
            raise AttributeError('Shape is already a tetrahedral mesh')
        
        return Shape(mo.to_tetrahedral(self))

    def to_triangular(self):
        """
        Get boundary triangle mesh of tetrahedra
        """
        if self.polyshape == 3:
            raise AttributeError('Shape is already a triangular mesh')
            
        return Shape(mo.to_triangular(self))
    
    ######################
    # Mesh/graph metrics #
    ######################
        
    def isoriented(self):
        """
        Check if tetrahedral mesh is oriented
        """
        return mo.isoriented(self)
        
    def average_edge_length(self):
        """
        Get average edge lengths
        """
        return mo.average_edge_length(self)
        
    def area(self):
        """
        Compute the total surface area of the mesh
        
        Returns
        -------
        float
            Total surface area.
        """
        return mo.area(self)
        
    def volume(self):
        """
        Compute the volume of closed triangular mesh or tetrahedral mesh
        
        Returns
        -------
        vol : float
            Total enclosed volume.
        """
        if self.closed is False:
            raise RuntimeError('Cannot compute volume, mesh is not closed')
        
        return mo.volume(self)
        
    def vertex_degrees(self):
        """
        Compute the vertex degrees (number of edges at each vertex)
        
        Returns
        -------
        vdeg : array
            Array of vertex degrees.
        """
        return mo.vertex_degrees(self)
        
    def vertex_areas(self):
        """
        Compute the area associated to each vertex 
        (1/3 of one-ring trias or 1/4 of one-ring tetras)
        
        Returns
        -------
        vareas : array
            Array of vertex areas.
        """
        return mo.vertex_areas(self)
        
    def vertex_normals(self):
        """
        Compute the vertex normals
        get_vertex_normals(v,t) computes vertex normals
            Normals around each vertex are averaged, weighted
            by the angle that they contribute.
            Ordering is important: counterclockwise when looking
            at the mesh from above.
            
        Returns
        -------
        n : array of shape (n_triangles, 3) or (n_tetrahedra, 4)
            Vertex normals.
        """
        return mo.vertex_normals(self)
        
    def boundary_loops(self):
        """Compute a tuple of boundary loops.

        Meshes can have 0 or more boundary loops, which are cycles in the directed
        adjacency graph of the boundary edges.
        Works on trias only. Could fail if loops are connected via a single
        vertex (like a figure 8). That case needs debugging.
        
        Returns
        -------
        loops : list of list
            List of lists with boundary loops.
        """
        return mo.boundary_loops(self)
        
    def centroid(self):
        """Compute centroid of mesh as a weighted average of triangle or 
        tetrahedra centers.

        The weight is determined by the triangle area.
        (This could be done much faster if a FEM lumped mass matrix M is
        already available where this would be M*v, because it is equivalent
        with averaging vertices weighted by vertex area)
        
        Returns
        -------
        centroid : float
            The centroid of the mesh.
        totalarea : float
            The total area of the mesh.
        """
        return mo.centroid(self)
        
    def curvature(self):
        """
        Compute various curvature values at vertices.

        .. note::

            For the algorithm see e.g.
            Pierre Alliez, David Cohen-Steiner, Olivier Devillers,
            Bruno Levy, and Mathieu Desbrun.
            Anisotropic Polygonal Remeshing.
            ACM Transactions on Graphics, 2003.

        Parameters
        ----------
        smoothit : int
            Smoothing iterations on vertex functions.

        Returns
        -------
        u_min : array of shape (vnum, 3)
            Minimal curvature directions.
        u_max : array of shape (vnum, 3)
            Maximal curvature directions.
        c_min : array
            Minimal curvature.
        c_max : array
            Maximal curvature.
        c_mean : array
            Mean curvature ``(c_min + c_max) / 2.0m``.
        c_gauss : array
           Gauss curvature ``c_min * c_maxm``.
        normals : array of shape (vnum, 3)
           Normals.
        """
        return mo.curvature(self)
        
    def shortest_path_length(self, source, target):
        """
        Calculates the shortest path length between two vertices in the mesh.

        Parameters
        ----------
        source : int
            Source vertex
        target : int
            Target vertex

        Returns
        -------
        float
            The shortest path length between source and target vertices. 
            Returns np.inf if no path exists.
        
        """
        return mo.shortest_path_length(self)
        
    def modularity(self): 
        """
        Calculates the Newman spectral modularity of the mesh.

        Returns
        -------
        float
            The Newman spectral modularity of the graph.
        """
        return mo.modularity(self)
        
    def geodesic_distance(self, v1, v2):
        """
        Find the geodesic distance of a vertex to another vertex
        """
        return mo.geodesic_distance(self, v1, v2)
        
    def geodesic_knn(self, vert_id, knn=100): 
        """
        Find the geodesic distances of a vertex to its k-nearest
        neighbors
        """
        return mo.geodesic_knn(self, knn)

    def geodesic_distmat(surf, n_jobs=1):
        """
        Compute geodesic distance using the heat diffusion method built into LaPy
            Based on: doi/10.1145/2516971.2516977
            Crane et al., "Geodesics in heat: A new approach to computing 
            distance based on heat flow"
        
        Parameters
        ----------
        surf : Shape class, must have Shape.polyshape == 3
            Input triangular surface
        n_jobs : int, optional
            Number of workers to use for parallel calls to 
            mesh.mesh_operations._distance_thread_method(),
            default is 1.
        
        Returns
        -------
        D : (num_verts, num_verts) np.memmap
            Geodesic distance matrix of every vertex to every other vertex
        
        """        
        return mo.geodesic_distmat(surf, n_jobs)
    
    def euclidean_distmat(self):
        """
        Compute euclidean distance matrix using `scipy.spatial.pdist`
        and `scipy.spatial.squareform`

        Returns
        -------
        D : (num_verts, num_verts) np.memmap
            Euclidean distance matrix of every vertex to every other vertex

        """        
        return mo.euclidean_distmat(self)
    
    def tria_normals(self):
        """
        Compute triangle normals
        """
        if self.polyshape == 4:
            raise AttributeError('Shape is tetrahedral, cannot compute triangular normals')
            
        return mo.tria_normals(self)
        
    def tria_qualities(self):
        """
        Compute triangle quality for each triangle
        """
        if self.polyshape == 4:
            raise AttributeError('Shape is tetrahedral, cannot compute triangle qualities')
            
        return mo.tria_qualities(self)
    
    def tria_areas(self):
        """Compute the area of triangles using Heron's formula.

        `Heron's formula <https://en.wikipedia.org/wiki/Heron%27s_formula>`_
        computes the area of a triangle by using the three edge lengths.

        Returns
        -------
        areas : array
            Array with areas of each triangle.
        """
        return mo.tria_areas(self)
    
    def is_manifold(self):
        """Check if triangle mesh is manifold (no edges with >2 triangles).

        Operates only on triangles
        
        Returns
        -------
        bool
            True if no edges with > 2 triangles.
        """
        if not self.polyshape == 3:
            raise AttributeError('Shape is tetrahedral, no manifold')
            
        return mo.is_manifold(self)
        
    def euler(self):
        """
        Compute the Euler Characteristic.

        The Euler characteristic is the number of vertices minus the number
        of edges plus the number of triangles  (= #V - #E + #T). For example,
        it is 2 for the sphere and 0 for the torus.
        This operates only on triangles array.
        
        Returns
        -------
        int
            Euler characteristic.
        """
        if self.polyshape == 4:
            raise AttributeError('Shape is tetrahedral, cannot compute euler characteristic')
            
        return mo.euler(self)
    
    def _construct_adj_sym(self):
        """
        Construct symmetric adjacency matrix (edge graph) of mesh
        
        Returns
        -------
        csc_matrix
            The non-directed adjacency matrix
            will be symmetric. Each inner edge (i,j) will have
            the number of triangles that contain this edge.
            Inner edges usually 2, boundary edges 1. Higher
            numbers can occur when there are non-manifold triangles.
            The sparse matrix can be binarized via:
                adj.data = np.ones(adj.data.shape).
        """
        return ma.compute_adj_matrix(self)
        
    ###################
    # LaPy attributes #
    ###################
    
    def compute_eigenmodes(self):
        """Calculate the eigenvalues and eigenmodes of a surface.

        Parameters
        ----------
        num_modes : int
            Number of eigenmodes to be calculated

        Returns
        ------
        evals : array (num_modes x 1)
            Eigenvalues
        emodes : array (number of surface points x num_modes)
            Eigenmodes
        """
        num_modes = self.num_modes
        self.evals, self.emodes = mo.calc_eigs(self.FEM, num_modes)
        
        return self.evals, self.emodes
    
    
    
    
    
    
    
    