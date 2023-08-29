"""
Mesh operation functions

References
----------
[MO1] Cohen 2013

[MO2] Crane et al., "Geodesics in heat: A new approach to computing distance 
    based on heat flow" `https://dx.doi.org/10.1145/2516971.2516977` 
"""
from scipy.spatial import cKDTree
from scipy.sparse import csgraph as csg
import scipy.sparse as sparse
from scipy.sparse.linalg import eigs
import numpy as np
import heapq
import sys
import inspect

from collections import deque

from lapy import Solver
from lapy.DiffGeo import tria_compute_gradient, tria_compute_divergence

from joblib import Parallel, delayed
from utils._imports import import_optional_dependency
import recon
import eigen
from . import data_operations as do
from . import iosupport as io
from . import mesh_io

# try import sksparse
try:
    from sksparse.cholmod import cholesky
except:
    print("Scikit-sparse libraries not found, using LU decomposition for eigenmodes (slower)")

import networkx as nx

norm_types = ['area', 'constant', 'number', 'volume']

#####################
# Vertex operations #
#####################

def sort_verts(surf, criterion='x'):
    """
    Returns sorted vertices
    
    Parameters
    ----------
    surf : Shape
        Mesh object on which vertices to sort
    criterion : str
        The sorting criterion. Can be 'x', 'y', 'z', 'distance', and 
        (if surf.data) 'data_weight'.
    
    Returns
    -------
    np.ndarray
        The sorted vertex coordinates
    """
    verts = surf.v
    if criterion == 'x':
        sorted_indices = np.argsort(verts[:, 0])
        
    elif criterion == 'y':
        sorted_indices = np.argsort(verts[:, 1])
        
    elif criterion == 'z':
        sorted_indices = np.argsort(verts[:, 2])
        
    elif criterion == 'distance':
        origin = np.zeros(3)
        distances = np.linalg.norm(verts - origin, axis=1)
        sorted_indices = np.argsort(distances)
        
    elif criterion == 'data_weight':
        if surf.data is None:
            raise AttributeError('Mesh must have data points')
        if surf.data.ndim > 1:
            raise AttributeError('Data is multi-dimensional, cannot sort by data')
        
        sorted_indices = np.argsort(surf.data)
        sorted_verts = verts[sorted_indices]
        
    else:
        raise ValueError("Invalid sorting criterion. Must be 'x', 'y', 'z', 'distance' or 'data_weight'.")

    return sorted_verts

def find_vertex_correspondence(surf, ref_surf, eps=0, n_jobs=1):
    """
    For each point in the input surface find its corresponding point
    in the reference surface.

    Parameters
    ----------
    surf : Shape
        Input mesh
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
    return _find_correspondence(surf, ref_surf, eps=eps, n_jobs=n_jobs,
                                use_faces=False)

def _find_correspondence(surf, ref_surf, eps=0, n_jobs=1):

    points = surf.get_verts
    ref_points = surf.get_verts

    tree = cKDTree(ref_points, leafsize=20, compact_nodes=False,
                   copy_data=False, balanced_tree=False)
    d, idx = tree.query(points, k=1, eps=0, n_jobs=n_jobs,
                        distance_upper_bound=eps+np.finfo(np.float64).eps)

    if np.isinf(d).any():
        raise ValueError('Cannot find correspondences. Try increasing '
                         'tolerance.')

    return idx

def vertex_neighbors(surf, vertex):
    """
    Returns the neighboring vertices of a given vertex.

    Parameters
    ----------
    surf : Shape
        Mesh object
    vertex : int
        Index of the vertex.

    Returns
    -------
    list
        List of neighboring vertex indices.
    """

    neighbors = set()

    for face in surf.f:
        if vertex in face:
            for vertex_index in face:
                if vertex_index != vertex:
                    neighbors.add(vertex_index)

    return list(neighbors)

def has_free_vertices(surf):
    """Check if the vertex list has more vertices than what is used.

    (same implementation as in `~lapy.TriaMesh`)
    
    Parameters
    ----------
    surf : Shape
        Mesh object

    Returns
    -------
    bool
        Whether vertex list has more vertices than what is used.
    """
    vnum = np.max(surf.v.shape)
    vnumt = len(np.unique(surf.f.reshape(-1)))
    return vnum != vnumt

def remove_free_vertices(surf):
    """Remove unused (free) vertices from v and t.

    These are vertices that are not used in any triangle. They can produce problems
    when constructing, e.g., Laplace matrices.

    Will update v and t in mesh.
    Same implementation as in `~lapy.TriaMesh`.
    
    Parameters
    ----------
    surf : Shape
        Mesh to remove free vertices from

    Returns
    -------
    vkeep: array
        Indices (from original list) of kept vertices.
    vdel: array
        Indices of deleted (unused) vertices.
    """
    if has_free_vertices(surf) is False:
        raise ValueError('Mesh has no free vertices')
    tflat = surf.f.reshape(-1)
    vnum = np.max(surf.v.shape)
    if np.max(tflat) >= vnum:
        raise ValueError("Max index exceeds number of vertices")
    # determine which vertices to keep
    vkeep = np.full(vnum, False, dtype=bool)
    vkeep[tflat] = True
    # list of deleted vertices (old indices)
    vdel = np.nonzero(~vkeep)[0]
    # if nothing to delete return
    if len(vdel) == 0:
        return np.arange(vnum), []
    # delete unused vertices
    vnew = surf.v[vkeep, :]
    # create lookup table
    tlookup = np.cumsum(vkeep) - 1
    # reindex tria
    tnew = tlookup[surf.f]
    # convert vkeep to index list
    vkeep = np.nonzero(vkeep)[0]
    surf.v = vnew
    surf.f = tnew
    return vkeep, vdel
    
def vertex_degrees(surf):
    """
    Compute the vertex degrees (number of edges at each vertex).

    Parameters
    ----------
    surf : Shape
        Mesh to compute vertex degrees.
    
    Returns
    -------
    vdeg : array
        Array of vertex degrees.
    """
    vdeg = np.bincount(surf.f.reshape(-1))
    return vdeg
    
def vertex_areas(surf):
    """
    Compute the area associated to each vertex 
    (1/3 of one-ring trias or 1/4 of one-ring tetras)
    
    Parameters
    ----------
    surf : Shape
        Mesh to compute one-ring vertex areas.
        
    Returns
    -------
    vareas : array
        Array of vertex areas.
    """
    if surf.polyshape == 3:
        v0 = surf.v[surf.f[:, 0], :]
        v1 = surf.v[surf.f[:, 1], :]
        v2 = surf.v[surf.f[:, 2], :]
        v1mv0 = v1 - v0
        v2mv0 = v2 - v0
        cr = np.cross(v1mv0, v2mv0)
        area = 0.5 * np.sqrt(np.sum(cr * cr, axis=1))
        area3 = np.repeat(area[:, np.newaxis], 3, 1)
        # varea = accumarray(t(:),area3(:))./3;
        vareas = np.bincount(surf.f.reshape(-1), area3.reshape(-1)) / 3.0
        return vareas
    
    if surf.polyshape == 4:
        vertex_areas = np.zeros(len(surf.v))

        for face in surf.f:
            v0 = surf.v[face[0]]
            v1 = surf.v[face[1]]
            v2 = surf.v[face[2]]
            v3 = surf.v[face[3]]

            # Calculate areas of three triangular faces
            area1 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            area2 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v3 - v0))
            area3 = 0.5 * np.linalg.norm(np.cross(v2 - v0, v3 - v0))
            area4 = 0.5 * np.linalg.norm(np.cross(v1 - v2, v3 - v2))

            # Add the areas to the corresponding vertices
            vertex_areas[face[0]] += (area1 + area2 + area3) / 3.0
            vertex_areas[face[1]] += (area1 + area4 + area2) / 3.0
            vertex_areas[face[2]] += (area1 + area4 + area3) / 3.0
            vertex_areas[face[3]] += (area2 + area4 + area3) / 3.0

        return vertex_areas
    
def vertex_normals(surf):
    """
    Compute the vertex normals
    get_vertex_normals(v,t) computes vertex normals
        Normals around each vertex are averaged, weighted
        by the angle that they contribute.
        Ordering is important: counterclockwise when looking
        at the triangle from above.
        
    Parameters
    ----------
    surf : Shape
        Mesh to compute vertex normals.
        
    Returns
    -------
    n : array of shape (n_triangles, 3) or (n_tetrahedra, 4)
        Vertex normals.
    """
    if not surf.is_oriented():
        raise ValueError(
            "Error: Vertex normals are meaningless for un-oriented meshes!"
        )
    import sys
    
    if surf.polyshape == 3:
        # Compute vertex coordinates and a difference vector for each triangle:
        v0 = surf.v[surf.f[:, 0], :]
        v1 = surf.v[surf.f[:, 1], :]
        v2 = surf.v[surf.f[:, 2], :]
        v1mv0 = v1 - v0
        v2mv1 = v2 - v1
        v0mv2 = v0 - v2
        # Compute cross product at every vertex
        # will all point in the same direction but have
        #   different lengths depending on spanned area
        cr0 = np.cross(v1mv0, -v0mv2)
        cr1 = np.cross(v2mv1, -v1mv0)
        cr2 = np.cross(v0mv2, -v2mv1)
        # Add normals at each vertex (there can be duplicate indices in t at vertex i)
        n = np.zeros(surf.v.shape)
        np.add.at(n, surf.f[:, 0], cr0)
        np.add.at(n, surf.f[:, 1], cr1)
        np.add.at(n, surf.f[:, 2], cr2)
        # Normalize normals
        ln = np.sqrt(np.sum(n * n, axis=1))
        ln[ln < sys.float_info.epsilon] = 1  # avoid division by zero
        n = n / ln.reshape(-1, 1)
        return n
    
    if surf.polyshape == 4:
        # Compute vertex coordinates and a difference vector for each tetrahedron:
        v0 = surf.v[surf.f[:, 0], :]
        v1 = surf.v[surf.f[:, 1], :]
        v2 = surf.v[surf.f[:, 2], :]
        v3 = surf.v[surf.f[:, 3], :]
        
        v1mv0 = v1 - v0
        v2mv0 = v2 - v0
        v3mv0 = v3 - v0
        v0mv2 = v0 - v2
        v0mv3 = v0 - v3

        cr0 = np.cross(v1mv0, -v0mv2)
        cr1 = np.cross(v2mv1, -v1mv0)
        cr2 = np.cross(v0mv2, -v2mv1)
        cr3 = np.cross(v0mv3, -v3mv0)

        n = np.zeros(surf.v.shape)
        np.add.at(n, surf.f[:, 0], cr0 + cr1 + cr3)
        np.add.at(n, surf.f[:, 1], cr0 + cr1 + cr2)
        np.add.at(n, surf.f[:, 2], cr0 + cr2 + cr3)
        np.add.at(n, surf.f[:, 3], cr1 + cr2 + cr3)

        ln = np.sqrt(np.sum(n * n, axis=1))
        ln[ln < sys.float_info.epsilon] = 1  # avoid division by zero
        n = n / ln.reshape(-1, 1)

        return n
    
def shortest_path_length(surf, source, target):
    """
    Calculates the shortest path length between two vertices in the mesh.

    Parameters
    ----------
    surf : Shape
        Mesh to compute shortest path length on
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
    adj = surf.adj_sym
    num_verts = surf.num_verts
    visited = set()
    queue = deque()

    queue.append(source)
    visited.add(source)

    distance = {node: np.inf for node in range(num_verts)}
    distance[source] = 0

    while queue:
        current_node = queue.popleft()
        visited.add(current_node)

        for neighbor in adj[:, current_node].nonzero()[0]:
            if neighbor not in visited:
                queue.append(neighbor)

                edge_weight = adj[neighbor, current_node]
                new_distance = distance[current_node] + edge_weight

                if new_distance < distance[neighbor]:
                    distance[neighbor] = new_distance

    return distance[target]
    
def geodesic_distance(surf, source, target):
    """
    Find the geodesic distance of a vertex to another vertex, uses heat diffusion
    method. Based on [MO2].
        
    Parameters
    ----------
    surf : Shape class
        Input surface
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
    if not surf.polyshape == 3:
        raise ValueError('Surface must be triangular to compute geodesic heat distance')
        
    if not surf.fem:
        surf.fem = Solver(surf, lump=True, use_cholmod=True)
        
    distances = diffusion(surf.TriaMesh, surf.fem, source)
    return distances[target]
    
def geodesic_knn(surf, source, m=1.0, knn=100): 
    """
    Find the geodesic distances of a vertex to its k-nearest
    neighbors using backward Euler solution in [MO2].
    
    Parameters
    ----------
    surf : Shape
        Mesh object
    source : int
        Source vertex
    m : float
        factor (default 1) to compute time of heat evolution: t = m * avg_edge_length^2
    knn : int
        Number of nearest neighbors to compute distances
        
    Returns
    -------
    ndarray of shape=(knn,)
        geodesic distances from the source vertex to its k-nearest neighbors
    """
    if not surf.polyshape == 3:
        raise ValueError('Surface must be triangular to compute geodesic heat distance')
    
    if not surf.fem:
        surf.fem = Solver(surf, lump=True, use_cholmod=True)
        
    sksparse = import_optional_dependency("sksparse", raise_error=True)
    
    nv = len(surf.v)
    fem = surf.fem
    # time of heat evolution:
    t = m * surf.avg_edge_length() ** 2
    # backward Euler matrix:
    hmat = fem.mass + t * fem.stiffness
    # set initial heat
    b0 = np.zeros((nv,))
    b0[np.array(source)] = 1.0
    # solve H x = b0
    if sksparse is not None:
        print("Solver: Cholesky decomposition from scikit-sparse cholmod ...")
        chol = sksparse.cholmod.cholesky(hmat)
        vfunc = chol(b0)   
        
    else:
        from scipy.sparse.linalg import splu
    
        print("Solver: spsolve (LU decomposition) ...")
        lu = splu(hmat)
        vfunc = lu.solve(np.float32(b0))
        
    # Calculate geodesic distances to k-nearest neighbors
    distances = []
    heap = []
    heapq.heappush(heap, (0, source))  # (distance, vertex_index)
    visited = set()
    
    while len(visited) < knn:
        if not heap:
            break
        dist, curr_vertex = heapq.heappop(heap)
        if curr_vertex not in visited:
            distances.append(dist)
            visited.add(curr_vertex)
            for neighbor in surf.vertex_neighbors(curr_vertex):
                if neighbor not in visited:
                    heapq.heappush(heap, (dist + vfunc[neighbor], neighbor))
    
    return distances

###################
# Face operations #
###################

# not implemented

###################
# Mesh operations #
###################

def apply_mask(surf, mask):
    """
    Mask surface vertices.

    Faces corresponding to these points are also kept.

    Parameters
    ----------
    surf : Shape
        Mesh object to mask
    mask : 1D ndarray or file-like
        Binary boolean array. Zero elements are discarded.

    Returns
    -------
    surf_masked : Shape
        Mesh object after masking

    """
    return _surface_mask(surf, mask)

def _surface_mask(surf, mask):
    """
    Mask vertices given binary array.

    Parameters
    ----------
    surf : Shape
        Mesh object to mask
    mask : str or ndarray
        Binary boolean or integer array. Zero or False elements are
        discarded.

    Returns
    -------
    surf_masked : Shape
        Mesh object after masking.

    """

    if isinstance(mask, np.ndarray):
        if np.issubdtype(mask.dtype, np.bool_):
            mask = mask.astype(np.uint8)
    else:
        try:
            mask = mesh_io.read_data(mask).astype(np.uint8)
        except Exception as e:
            raise RuntimeError(e)

    if np.any(np.unique(mask) > 1):
        raise ValueError('Cannot work with non-binary mask.')

    num_verts = surf.num_verts
    num_faces = surf.num_faces
    vertices = surf.v
    faces = surf.f
    
    
    # Create an inverse mask to keep non-masked vertices
    inverse_mask = np.logical_not(mask)

    # Create a mapping from old vertex indices to new vertex indices
    vertex_mapping = np.cumsum(inverse_mask) - 1

    # Update the vertex coordinates based on the mask
    updated_vertices = vertices[inverse_mask]
    
    # Update the face indices based on the mask
    updated_faces = np.zeros_like(faces)
    for i in range(num_faces):
        for j in range(3):
            old_vertex_index = faces[i, j]
            if mask[old_vertex_index]:
                updated_faces[i, j] = vertex_mapping[old_vertex_index]
            else:
                updated_faces[i, j] = old_vertex_index
                
    updated_data_arrays = None
    if surf.data is not None:
        updated_data_arrays = []
        for data_array in surf.data:
            updated_data_arrays.append(data_array[inverse_mask])

    return updated_vertices, updated_faces, updated_data_arrays

def threshold_surface(surf, threshold_type='distance', threshold=0.1):
    """
    Selection of edges in mesh adjacency matrix based on `threshold_type`
    and `threshold`.

    Parameters
    ----------
    surf : Shape object
        Surface to threshold.
    threshold_type : str, optional
        Thresholding type. The default is 'distance'.
    threshold : float, optional
        Weight to threshold based on `threshold_type`. The default is 0.1.

    Raises
    ------
    ValueError
        `threshold_type` not in `mesh.mesh_operations.threshold_types`

    Returns
    -------
    thresholded_adj : csc_matrix
        Thresholded adjacency matrix
        
    Notes
    -----
    See `mesh.mesh_operations.threshold_types`

    """
    threshold_types = ['distance', 'variance', 'strength']
    
    if not threshold_type in threshold_types:
        raise ValueError("Threshold type must be 'distance', 'variance', 'strength'")
    
    adj = surf.adj_sym
    
    if threshold_type == 'distance':
        thresholded_adj = adj.copy()
        thresholded_adj[thresholded_adj < threshold] = 0

    elif threshold_type == 'variance':
        # Calculate the variance of edge weights
        edge_weights = adj.data
        variance = np.var(edge_weights)

        # Threshold based on variance
        thresholded_adj = adj.copy()
        thresholded_adj[thresholded_adj < variance * threshold] = 0

    elif threshold_type == 'strength':
        # Threshold based on edge weight strength
        thresholded_adj = adj.copy()
        thresholded_adj[thresholded_adj < threshold] = 0
    
    mask = np.array(thresholded_adj.sum(axis=1)).flatten() > 0

    return apply_mask(surf, mask)

def threshold_minimum(surf, return_threshold=True):
    """
    Threshold surface adjacency matrix by minimum value needed to 
    remain connected. Based on [MO1].

    Parameters
    ----------
    surf : Shape object
        Surface to threshold.
    return_threshold : bool
        Returns minimum threshold if "True"

    Returns
    -------
    thresholded_surface : Shape
        Thresholded surface
    min_threshold : float, optional
        If `return_threshold` is "True", returns minimum threshold needed to
        remain fully connected.
    
    References
    ----------
    [MO1] Cohen 2013

    """
    # Get the connected components of the original mesh
    adj = surf.e    
    _, original_labels = csg.connected_components(adj, directed=False)

    # Sort the unique labels by their sizes
    unique_labels, label_counts = np.unique(original_labels, return_counts=True)
    sorted_labels = unique_labels[np.argsort(label_counts)]

    # Threshold the adjacency matrix incrementally to find the minimum threshold
    min_threshold = 0
    for i in range(1, len(sorted_labels)):
        mask = original_labels < sorted_labels[i]
        thresholded_adj = adj.copy()
        thresholded_adj[mask, :] = 0
        thresholded_adj[:, mask] = 0

        _, labels = csg.connected_components(thresholded_adj, directed=False)
        if len(np.unique(labels)) > 1:
            break
        min_threshold = thresholded_adj.data.max()
    
    mask = np.array(thresholded_adj.sum(axis=1)).flatten() > 0
    
    if return_threshold is True:
        return apply_mask(surf, mask), min_threshold
    
    return apply_mask(surf, mask)
    
def combine_surfaces(surf, *new_surfaces):
    """
    Combine surfaces given sequence of polydata or Shape class. Accepts
    file_likes as new surfaces to combine.
    
    Parameters
    ----------
    surf : Shape
        Original surface
    *new_surfaces : list
        Meshes to combine with original surface. Accepts str of file_like, 
        np.ndarray, or list of lists of vertices and faces.
    
    Returns
    -------
    Shape
        Combined surfaces
    """
    combined_vertices = []
    combined_faces = []
    combined_data = None
    if surf.data:
        combined_data = []

    for surface in new_surfaces:
        if io.is_string_like(surface):
            try:
                surface = io.load(surface)
            except Exception as e:
                raise RuntimeWarning(e)
                continue
        if surface.__name__:
            vertices, faces = surface.v, surface.f
            if surface.data:
                data = surface.data
        else:
            try:
                vertices, faces, data = surface
            except:
                raise RuntimeWarning('Could not load surface, check input')
                continue
            
        num_vertices = len(combined_vertices)

        # Update vertex indices of the current mesh
        updated_faces = faces + num_vertices

        # Append vertex coordinates and face indices to the combined lists
        combined_vertices.extend(vertices)
        combined_faces.extend(updated_faces)
        if data is not None:
            combined_data.extend(data)
        else:
            # Add None for missing data
            combined_data.extend([None] * len(vertices))

    return np.array(combined_vertices), np.array(combined_faces), np.array(combined_data)

def split_surface(surf, labels):
    """
    Split surface into separate meshes based on a label array.
    
    Parameters
    ----------
    surf : Shape
        The mesh to split into separate surfaces
    labels : ndarray or str of file_like
        Label array of ints specifying the label for each vertex.
        
    Returns
    -------
    split_surfaces : list
        List where each entry is a separate Shape object split according
        to the labels
    """
    if not isinstance(labels, np.ndarray):
        labels = io.load(labels) # let io handle
    if np.issubdtype(labels.dtype, np.bool_):
        labels = labels.astype(np.uint8)
    if np.issubdtype(labels.dtype, np.uint8):
        raise ValueError('Label array must be integer array')
        
    unique_labels = np.unique(labels)

    split_surfaces = []
    masked_data = None
    vertices = surf.v
    faces = surf.f

    for label in unique_labels:
        mask = labels == label
        masked_vertices = vertices[mask]
        masked_faces = faces[np.isin(faces, np.where(mask)[0]).any(axis=1)]
        
        if surf.data is not None:
            masked_data = surf.data[mask]
        
        mesh = masked_vertices, masked_faces, masked_data
        split_surfaces.append(mesh)

    return split_surfaces

def normalize(surf, mask=None, norm_type='volume', norm_factor=1.0):
    """
    Normalize mesh by `norm_type` in `norm_types`. Normalization types listed
    in `mesh.mesh_operations.norm_types`. Can also normalize to given mask.
    
    Parameters
    ----------
    surf : Shape
        Mesh to normalize
    mask : ndarray of bool or binary labels (0, 1)
        
    norm_type : str
        Normalization type. Accepted inputs are 'area', 'constant', 'number',
        or 'volume'. Default is 'area'.
        number - normalization with respect to the total number of vertices
        volume - normalization with respect to the total volume of mesh
        constant - normalization with respect to chosen constant in `norm_factor`
        area - normalization with respect to total surface area
    norm_factor : float
        Normalization factor only used in 'constant'.
        
    Returns
    -------
    surf_norm : Shape
        Normalized mesh
    """
    if norm_type not in norm_types:
        raise ValueError("Normalization types are 'area', 'constant', 'number', or 'volume'. Got {}".format(norm_type))
    
    surf_norm = surf
    
    factor = 3 if surf.polyshape == 3 else 4
    
    if mask:
        surf_masked = apply_mask(surf, mask)
        if norm_type == 'number':
            surf_norm.v = surf.v/(surf_masked.v**(1/factor))
        elif norm_type == 'volume':
            surf_norm.v = surf.v/(volume(surf_masked)**(1/factor))
        elif norm_type == 'constant':
            surf_norm.v = surf.v/(norm_factor**(1/factor))
        elif norm_type == 'area':
            surf_norm.v = surf.v/(area(surf_masked)**(1/factor))
        return surf_norm
    
    if norm_type == 'number':
        surf_norm.v = surf.v/(surf.v**(1/factor))
    elif norm_type == 'volume':
        surf_norm.v = surf.v/(volume(surf)**(1/factor))
    elif norm_type == 'constant':
        surf_norm.v = surf.v/(norm_factor**(1/factor))
    elif norm_type == 'area':
        surf_norm.v = surf.v/(area(surf)**(1/factor))
        
    return surf_norm
    
def unit_normalize(surf):
    """
    Normalize mesh to unit surface area and centroid at origin. Same as 
    `~lapy.TriaMesh()` method without modifying in-place.
    
    Parameters
    ----------
    surf : Shape
        Mesh to unit normalize
    
    Returns
    -------
    surf_norm : Shape
        Normalized mesh
    
    """
    surf_norm = surf
    cent, area = centroid(surf)
    surf_norm.v = (1.0 / np.sqrt(area)) * (surf_norm.v - cent)
    
    return surf_norm
    
def refine(surf, it=1):
    """
    Refine the mesh by placing new vertex on each edge midpoint. Creates 4
    similar triangles (if triangular) or 8 similar tetrahedra (if tetrahedral)
    from one parent. Same as `~lapy.TriaMesh()` method without modifying in-place
    for tria mesh. Uses Doo-Sabin algorithm for tetrahedral mesh.
    
    Does not interpolate data on the new surface.
    
    Parameters
    ----------
    surf : Shape
        Mesh to refine
    it : int
        Number of times to repeat the refining process
    
    Returns
    -------
    surf_refined : Shape
        Refined surface with 4-8 times the number of faces
        
    :warning: Refining a mesh with more than 100k vertices can quickly overflow!
    Use at your own risk!
    """
    
    if surf.polyshape == 3:
        for x in range(it):
            # make symmetric adj matrix to upper triangle
            adjtriu = sparse.triu(surf.adj_sym, 0, format="csr")
            # create new vertex index for each edge
            edgeno = adjtriu.data.shape[0]
            vno = surf.v.shape[0]
            adjtriu.data = np.arange(vno, vno + edgeno)
            # get vertices at edge midpoints:
            rows, cols = adjtriu.nonzero()
            vnew = 0.5 * (surf.v[rows, :] + surf.v[cols, :])
            vnew = np.append(surf.v, vnew, axis=0)
            # make adj symmetric again
            adjtriu = adjtriu + adjtriu.f
            # create 4 new triangles for each old one
            e1 = np.asarray(adjtriu[surf.f[:, 0], surf.f[:, 1]].flat)
            e2 = np.asarray(adjtriu[surf.f[:, 1], surf.f[:, 2]].flat)
            e3 = np.asarray(adjtriu[surf.f[:, 2], surf.f[:, 0]].flat)
            t1 = np.column_stack((surf.f[:, 0], e1, e3))
            t2 = np.column_stack((surf.f[:, 1], e2, e1))
            t3 = np.column_stack((surf.f[:, 2], e3, e2))
            t4 = np.column_stack((e1, e2, e3))
            tnew = np.reshape(np.concatenate((t1, t2, t3, t4), axis=1), (-1, 3))
            # set new vertices and tria and re-init adj matrices
            surf_refined = vnew, tnew
            return surf_refined
    
    if surf.polyshape == 4:
        vnew = surf.v.copy()
        tnew = surf.f.copy()
    
        for _ in range(it):
            num_vertices = len(vnew)
            num_tetrahedra = len(tnew)
    
            # Create containers for new vertices and tetrahedra
            subdivided_vertices = []
            subdivided_tetrahedra = []
    
            for tetra_idx in range(num_tetrahedra):
                tetra = tnew[tetra_idx]
    
                # Calculate the midpoints of the tetrahedron edges
                midpoints = []
                for i in range(4):
                    for j in range(i + 1, 4):
                        midpoint = 0.5 * (vnew[tetra[i]] + vnew[tetra[j]])
                        midpoints.append(midpoint)
    
                # Calculate the average of the tetrahedron vertices
                avg_vertex = np.mean(vnew[tetra], axis=0)
    
                # Create new vertices
                subdivided_vertices.extend(midpoints)
                subdivided_vertices.append(avg_vertex)
    
                # Create new tetrahedra using the new vertices
                v1, v2, v3, v4, v5, v6, v7 = (
                    tetra[0],
                    tetra[1],
                    tetra[2],
                    tetra[3],
                    num_vertices + tetra_idx * 6,
                    num_vertices + tetra_idx * 6 + 1,
                    num_vertices + tetra_idx * 6 + 2,
                )
    
                subdivided_tetrahedra.append([v1, v5, v6, v7])
                subdivided_tetrahedra.append([v2, v7, v6, v4])
                subdivided_tetrahedra.append([v3, v4, v6, v5])
                subdivided_tetrahedra.append([v5, v6, v7, v4])
    
            vnew = np.array(subdivided_vertices)
            tnew = np.array(subdivided_tetrahedra)
        
        surf_refined = vnew, tnew
        return surf_refined
    
def orient(surf):
    """
    Re-orient mesh to be consistent.

    If triangular:
        Re-orients triangles of manifold mesh to be consistent, so that vertices are
        listed counter-clockwise, when looking from above (outside).
    
        Algorithm:
    
        * Construct list for each half-edge with its triangle and edge direction
        * Drop boundary half-edges and find half-edge pairs
        * Construct sparse matrix with triangle neighbors, with entry 1 for opposite
          half edges and -1 for parallel half-edges (normal flip across this edge)
        * Flood mesh from first tria using triangle neighbor matrix and keeping track of
          sign
        * When flooded, negative sign for a triangle indicates it needs to be flipped
        * If global volume is negative, flip everything (first tria was wrong)
    
    If tetrahedral:
        

    Returns
    -------
    flipped : int
        Number of trias flipped.
    """
    if surf.polyshape == 3:
        tnew = surf.f
        flipped = 0
        if not surf.is_oriented():
            # get half edges
            t0 = surf.f[:, 0]
            t1 = surf.f[:, 1]
            t2 = surf.f[:, 2]
            # i,j are beginning and end points of each half edge
            i = np.column_stack((t0, t1, t2)).reshape(-1)
            j = np.column_stack((t1, t2, t0)).reshape(-1)
            # tidx for each half edge
            tidx = np.repeat(np.arange(0, surf.f.shape[0]), 3)
            # if edge points from smaller to larger index or not
            dirij = i < j
            ndirij = np.logical_not(dirij)
            ij = np.column_stack((i, j))
            # make sure i < j
            ij[np.ix_(ndirij, [1, 0])] = ij[np.ix_(ndirij, [0, 1])]
            # remove rows with unique (boundary) edges (half-edges without partner)
            u, ind, c = np.unique(ij, axis=0, return_index=True, return_counts=True)
            bidx = ind[c == 1]
            # assert remaining edges have two triangles: min = max =2
            # note if we have only a single triangle or triangle soup
            # this will fail as we have no inner edges.
            if max(c) != 2 or min(c) < 1:
                raise ValueError(
                    "Without boundary edges, all should have two triangles!"
                )
            # inner is a mask for inner edges
            inner = np.ones(ij.shape[0], bool)
            inner[bidx] = False
            # stack i,j,tria_id, edge_direction (smaller to larger vidx) for inner edges
            ijk = np.column_stack((ij, tidx, dirij))[inner, :]
            # sort according to first two columns
            ind = np.lexsort(
                (ijk[:, 0], ijk[:, 1])
            )  # Sort by column 0, then by column 1
            ijks = ijk[ind, :]
            # select both tria indices at each edge and the edge directions
            tdir = ijks.reshape((-1, 8))[:, [2, 6, 3, 7]]
            # compute sign vector (1 if edge points from small to large, else -1)
            tsgn = 2 * np.logical_xor(tdir[:, 2], tdir[:, 3]) - 1
            # append to itsurf for symmetry
            tsgn = np.append(tsgn, tsgn)
            i = np.append(tdir[:, 0], tdir[:, 1])
            j = np.append(tdir[:, 1], tdir[:, 0])
            # construct sparse tria neighbor matrix where
            #   weights indicate normal flips across edge
            tmat = sparse.csc_matrix((tsgn, (i, j)))
            tdim = max(i) + 1
            tmat = tmat + sparse.eye(tdim)
            # flood by starting with neighbors of tria 0 to fill all trias
            # sadly we still need a loop for this, matrix power would be too slow
            # as we don't really need to compute full matrix, only need first column
            v = tmat[:, 0]
            count = 0
            import time
    
            startt = time.time()
            while len(v.data) < tdim:
                count = count + 1
                v = tmat * v
                v.data = np.sign(v.data)
            endt = time.time()
            print(
                "Searched mesh after {} flooding iterations ({} sec).".format(
                    count, endt - startt
                )
            )
            # get tria indices that need flipping:
            idx = v.toarray() == -1
            idx = idx.reshape(-1)
            tnew = surf.f
            tnew[np.ix_(idx, [1, 0])] = tnew[np.ix_(idx, [0, 1])]
            surf.__init__(surf.v, tnew)
            flipped = idx.sum()
        # flip orientation on all trias if volume is negative:
        if surf.volume() < 0:
            tnew[:, [1, 2]] = tnew[:, [2, 1]]
            surf.__init__(surf.v, tnew)
            flipped = tnew.shape[0] - flipped
        return flipped
    
    #if surf.polyshape == 4:
        
    
# def smooth:
#     """
#     Smooth the mesh iteratively with kernels in `kernel_types`
#     """
    
    
def decimate(surf, decimation_factor=0.1):
    """
    Decimation of vertices and faces according to decimation factor
    
    Parameters
    ----------
    surf : Shape
        Mesh to decimate
    decimation_factor : float
        Value between 0 and 1. For example, 0.5 retains half the 
        original vertices.
        
    Returns
    -------
    decimated_surface : Shape
        Decimated mesh
    """
    num_verts = surf.num_verts
    num_faces = surf.num_faces
    vertices = surf.v
    faces = surf.f
    
    # Calculate the number of vertices to retain
    num_vertices_retain = int(num_verts * decimation_factor)

    # Construct a KDTree for efficient nearest neighbor search
    kdtree = cKDTree(vertices)

    # Compute the average distance to the nearest neighbor for each vertex
    _, avg_distances = kdtree.query(vertices, k=2)
    avg_distances = np.mean(avg_distances[:, 1:], axis=1)

    # Sort the vertices based on the average distances
    sorted_indices = np.argsort(avg_distances)

    # Retain the first num_vertices_retain vertices
    decimated_vertices = vertices[sorted_indices[:num_vertices_retain]]

    # Update the face indices based on the retained vertices
    decimated_faces = np.zeros_like(faces)
    for i in range(num_faces):
        for j in range(3):
            old_vertex_index = faces[i, j]
            new_vertex_index = np.where(sorted_indices == old_vertex_index)[0][0]
            decimated_faces[i, j] = new_vertex_index
            
    # decimated data
    decimated_data = None
    if surf.data:
        decimated_data = surf.data[sorted_indices[:num_vertices_retain]]
    
    return decimated_vertices, decimated_faces, decimated_data
    
def decimate_pro(surf, decimation_factor=0.1):
    """
    Decimation of vertices and faces according to decimation factor,
    uses Pyvista `decimate_pro` method
    
    Parameters
    ----------
    surf : Shape
        Mesh to decimate
    decimation_factor : float
        Value between 0 and 1. For example, 0.5 retains half the 
        original vertices. 
    """
    import pyvista as pv
    
    vertices = surf.v
    faces = surf.f
    
    # Create a PyVista mesh from the vertex coordinates and face indices
    mesh = pv.PolyData(vertices, faces)
    if surf.data:
        mesh.point_arrays['Data'] = surf.data

    # Perform decimation using the decimate_pro method
    decimated_mesh = mesh.decimate_pro(target_reduction=1 - decimation_factor)
    decimated_data = None

    # Extract the decimated vertex coordinates and face indices from the decimated mesh
    decimated_vertices = decimated_mesh.points
    if surf.polyshape == 3:
        decimated_faces = decimated_mesh.faces.reshape(-1, 3)[:, 1:]
    if surf.polyshape == 4:
        decimated_faces = decimated_mesh.faces.reshape(-1, 4)[:, 1:]
    if surf.data:
        decimated_data = decimated_mesh.point_arrays['Data']
        
    return decimated_vertices, decimated_faces, decimated_data 
    
######################
# Mesh/graph metrics #
######################
    
def get_connected_components(surf):
    """
    Get connected components
    """
    adj = surf.e
    
    return csg.connected_components(adj, directed=False, connection='weak')
    
def isoriented(surf):
    """Check if mesh is oriented.

    True if all faces are oriented
    so that v0,v1,v2 are oriented counterclockwise when looking from above,
    and v3 is on top of that triangle.

    Returns
    -------
    oriented: bool
        True if ``max(adj_directed)=1``.
    """
    
    if surf.polyshape == 3:
        """Check if triangle mesh is oriented.
        
        True if all triangles are oriented counter-clockwise, when looking from
        above. Operates only on triangles.
        
        Returns
        -------
        bool
        True if ``max(adj_directed)=1``.
        """
        return np.max(surf.adj_dir.data) == 1
   
    else:
        # Compute vertex coordinates and a difference vector for each triangle:
        t0 = surf.f[:, 0]
        t1 = surf.f[:, 1]
        t2 = surf.f[:, 2]
        t3 = surf.f[:, 3]
        v0 = surf.v[t0, :]
        v1 = surf.v[t1, :]
        v2 = surf.v[t2, :]
        v3 = surf.v[t3, :]
        e0 = v1 - v0
        e2 = v2 - v0
        e3 = v3 - v0
        # Compute cross product and 6 * vol for each triangle:
        cr = np.cross(e0, e2)
        vol = np.sum(e3 * cr, axis=1)
        if np.max(vol) < 0.0:
            print("All tet orientations are flipped")
            return False
        elif np.min(vol) > 0.0:
            print("All tet orientations are correct")
            return True
        elif np.count_nonzero(vol) < len(vol):
            print("We have degenerated zero-volume tetrahedra")
            return False
        else:
            print("Orientations are not uniform")
            return False
    
def average_edge_length(surf):
    """Get average edge lengths in mesh.

    Returns
    -------
    float
        Average edge length.
    """
    # get only upper off-diag elements from symmetric adj matrix
    triadj = sparse.triu(surf.adj_sym, 1, format="coo")
    edgelens = np.sqrt(
        ((surf.v[triadj.row, :] - surf.v[triadj.col, :]) ** 2).sum(1)
    )
    return edgelens.mean()
    
def area(surf):
    """Compute the total surface area of the mesh.
    
    Parameters
    ----------
    surf : Shape
        Triangular or tetrahedral surface to compute surface area.

    Returns
    -------
    float
        Total surface area.
    """
    
    if surf.polyshape == 3:
        areas = surf.tria_areas()
        return np.sum(areas)
    
    if surf.polyshape == 4:
        tria = to_triangular(surf)
        areas = tria_areas(surf)
        return np.sum(areas)
    
def volume(surf):
    """Compute the volume of closed mesh, summing at origin.
    
    Parameters
    ----------
    surf : Shape
        Mesh to compute volume.

    Returns
    -------
    vol : float
        Total enclosed volume.
    """
    if not surf.closed:
        return AttributeError('Mesh is not closed, cannot compute volume')
    if not surf.is_oriented():
        raise ValueError(
            "Error: Can only compute volume for oriented meshes!"
        )
        
    if surf.polyshape == 3:
        v0 = surf.v[surf.f[:, 0], :]
        v1 = surf.v[surf.f[:, 1], :]
        v2 = surf.v[surf.f[:, 2], :]
        v1mv0 = v1 - v0
        v2mv0 = v2 - v0
        cr = np.cross(v1mv0, v2mv0)
        spatvol = np.sum(v0 * cr, axis=1)
        vol = np.sum(spatvol) / 6.0
        return vol
    
    if surf.polyshape == 4:
        v0 = surf.v[surf.f[:, 0], :]
        v1 = surf.v[surf.f[:, 1], :]
        v2 = surf.v[surf.f[:, 2], :]
        v3 = surf.v[surf.f[:, 3], :]

        v1mv0 = v1 - v0
        v2mv0 = v2 - v0
        v3mv0 = v3 - v0

        cr = np.cross(v1mv0, v2mv0)
        spatvol = np.sum(v0 * cr, axis=1)

        vol = np.sum(v3mv0 * spatvol) / 8.0
        return vol
    

    
def boundary_loops(surf):
    """Compute a tuple of boundary loops.

    Meshes can have 0 or more boundary loops, which are cycles in the directed
    adjacency graph of the boundary edges.
    Works on trias only. Could fail if loops are connected via a single
    vertex (like a figure 8). That case needs debugging.
    
    Parameters
    ----------
    surf : Shape
        Mesh to compute boundary loops.

    Returns
    -------
    loops : list of list
        List of lists with boundary loops.
    """
    if not surf.polyshape == 3:
        raise AttributeError('Mesh must be triangular to compute boundary loops')
    if not surf.is_manifold():
        raise ValueError(
            "Error: tria not manifold (edges with more than 2 triangles)!"
        )
    if surf.is_closed():
        return []
    # get directed matrix of only boundary edges
    inneredges = surf.adj_sym == 2
    if not surf.is_oriented():
        raise ValueError("Error: tria not oriented !")
    adj = surf.adj_dir.copy()
    adj[inneredges] = 0
    adj.eliminate_zeros()
    # find loops
    # get first column index with an entry:
    firstcol = np.nonzero(adj.indptr)[0][0] - 1
    loops = []
    # loop while we have more first columns:
    while not firstcol == []:
        # start the new loop with this index
        loop = [firstcol]
        # delete this entry from matrix (visited)
        adj.data[adj.indptr[firstcol]] = 0
        # get the next column (=row index of the first entry (and only, hopefully)
        ncol = adj.indices[adj.indptr[firstcol]]
        # as long as loop is not closed walk through it
        while not ncol == firstcol:
            loop.append(ncol)
            adj.data[adj.indptr[ncol]] = 0  # visited
            ncol = adj.indices[adj.indptr[ncol]]
        # get rid of the visited nodes, store loop and check for another one
        adj.eliminate_zeros()
        loops.append(loop)
        nz = np.nonzero(adj.indptr)[0]
        if len(nz) > 0:
            firstcol = nz[0] - 1
        else:
            firstcol = []
    return loops

def centroid(surf):
    """Compute centroid of mesh as a weighted average of triangle or 
    tetrahedra centers.

    The weight is determined by the triangle area.
    (This could be done much faster if a FEM lumped mass matrix M is
    already available where this would be M*v, because it is equivalent
    with averaging vertices weighted by vertex area)
    
    Parameters
    ----------
    surf : Shape
        Mesh to compute centroid.

    Returns
    -------
    centroid : float
        The centroid of the mesh.
    totalarea : float
        The total area of the mesh.
    """
    
    if surf.polyshape == 3:
        v0 = surf.v[surf.f[:, 0], :]
        v1 = surf.v[surf.f[:, 1], :]
        v2 = surf.v[surf.f[:, 2], :]
        v2mv1 = v2 - v1
        v0mv2 = v0 - v2
        # Compute cross product and area for each triangle:
        cr = np.cross(v2mv1, v0mv2)
        areas = 0.5 * np.sqrt(np.sum(cr * cr, axis=1))
        totalarea = areas.sum()
        areas = areas / totalarea
        centers = (1.0 / 3.0) * (v0 + v1 + v2)
        c = centers * areas[:, np.newaxis]
        return np.sum(c, axis=0), totalarea
    
    if surf.polyshape == 4:
        v0 = surf.v[surf.f[:, 0], :]
        v1 = surf.v[surf.f[:, 1], :]
        v2 = surf.v[surf.f[:, 2], :]
        v3 = surf.v[surf.f[:, 3], :]
    
        v1mv0 = v1 - v0
        v2mv0 = v2 - v0
        v3mv0 = v3 - v0
    
        cr0 = np.cross(v1mv0, v2mv0)
        cr1 = np.cross(v1mv0, v3mv0)
        cr2 = np.cross(v2mv0, v3mv0)
        cr3 = np.cross(v1mv0, v2mv0)
    
        areas = (np.linalg.norm(cr0, axis=1) + np.linalg.norm(cr1, axis=1) + np.linalg.norm(cr2, axis=1) + np.linalg.norm(cr3, axis=1)) / 6.0
        totalarea = areas.sum()
        areas = areas / totalarea
    
        centers = (1.0 / 4.0) * (v0 + v1 + v2 + v3)
        c = centers * areas[:, np.newaxis]
    
        return np.sum(c, axis=0), totalarea
    
def curvature(surf, smoothing_iterations=3):
    """
    Compute various curvature values at vertices. Based on ~lapy.TriaMesh

    .. note::

        For the algorithm see e.g.
        Pierre Alliez, David Cohen-Steiner, Olivier Devillers,
        Bruno Levy, and Mathieu Desbrun.
        Anisotropic Polygonal Remeshing.
        ACM Transactions on Graphics, 2003.

    Parameters
    ----------
    surf : Shape
        Mesh to compute curvature. Must be triangular.
    smoothing_iterations : int
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
    if surf.polyshape == 4:
        raise AttributeError('Shape is tetrahedral, cannot compute curvature')
    
    # import warnings
    # warnings.filterwarnings('error')

    # get edge information for inner edges (vertex ids and tria ids):
    vids, tids = surf.edges()
    # compute normals for each tria
    tnormals = surf.tria_normals()
    # compute dot product of normals at each edge
    sprod = np.sum(tnormals[tids[:, 0], :] * tnormals[tids[:, 1], :], axis=1)
    # compute unsigned angles (clamp to ensure range)
    angle = np.maximum(sprod, -1)
    angle = np.minimum(angle, 1)
    angle = np.arccos(angle)
    # compute edge vectors and lengths
    edgevecs = surf.v[vids[:, 1], :] - surf.v[vids[:, 0], :]
    edgelen = np.sqrt(np.sum(edgevecs**2, axis=1))
    # get sign (if normals face towards each other or away, across each edge)
    cp = np.cross(tnormals[tids[:, 0], :], tnormals[tids[:, 1], :])
    si = -np.sign(np.sum(cp * edgevecs, axis=1))
    angle = angle * si
    # normalized edges
    edgelen[edgelen < sys.float_info.epsilon] = 1  # avoid division by zero
    edgevecs = edgevecs / edgelen.reshape(-1, 1)
    # adjust edgelengths so that mean is 1 for numerics
    edgelen = edgelen / np.mean(edgelen)
    # symmetric edge matrix (3x3, upper triangular matrix entries):
    ee = np.empty([edgelen.shape[0], 6])
    ee[:, 0] = edgevecs[:, 0] * edgevecs[:, 0]
    ee[:, 1] = edgevecs[:, 0] * edgevecs[:, 1]
    ee[:, 2] = edgevecs[:, 0] * edgevecs[:, 2]
    ee[:, 3] = edgevecs[:, 1] * edgevecs[:, 1]
    ee[:, 4] = edgevecs[:, 1] * edgevecs[:, 2]
    ee[:, 5] = edgevecs[:, 2] * edgevecs[:, 2]
    # scale angle by edge lengths
    angle = angle * edgelen
    # multiply scaled angle with matrix entries
    ee = ee * angle.reshape(-1, 1)
    # map to vertices
    vnum = surf.v.shape[0]
    vv = np.zeros([vnum, 6])
    np.add.at(vv, vids[:, 0], ee)
    np.add.at(vv, vids[:, 1], ee)
    vdeg = np.zeros([vnum])
    np.add.at(vdeg, vids[:, 0], 1)
    np.add.at(vdeg, vids[:, 1], 1)
    # divide by vertex degree (maybe better by edge length sum??)
    vdeg[vdeg == 0] = 1
    vv = vv / vdeg.reshape(-1, 1)
    # smooth vertex functions
    vv = smooth(vv, smoothing_iterations)
    # create vnum 3x3 symmetric matrices at each vertex
    mats = np.empty([vnum, 3, 3])
    mats[:, 0, :] = vv[:, [0, 1, 2]]
    mats[:, [1, 2], 0] = vv[:, [1, 2]]
    mats[:, 1, [1, 2]] = vv[:, [3, 4]]
    mats[:, 2, 1] = vv[:, 4]
    mats[:, 2, 2] = vv[:, 5]
    # compute eigendecomposition (real for symmetric matrices)
    evals, evecs = np.linalg.eig(mats)
    evals = np.real(evals)
    evecs = np.real(evecs)
    # sort evals ascending
    # this is instable in perfectly planar regions
    #  (normal can lie in tangential plane)
    # i = np.argsort(np.abs(evals), axis=1)
    # instead we find direction that aligns with vertex normals as first
    # the other two will be sorted later anyway
    vnormals = surf.vertex_normals()
    dprod = -np.abs(np.squeeze(np.sum(evecs * vnormals[:, :, np.newaxis], axis=1)))
    i = np.argsort(dprod, axis=1)
    evals = np.take_along_axis(evals, i, axis=1)
    it = np.tile(i.reshape((vnum, 1, 3)), (1, 3, 1))
    evecs = np.take_along_axis(evecs, it, axis=2)
    # pull min and max curv. dirs
    u_min = np.squeeze(evecs[:, :, 2])
    u_max = np.squeeze(evecs[:, :, 1])
    c_min = evals[:, 1]
    c_max = evals[:, 2]
    normals = np.squeeze(evecs[:, :, 0])
    c_mean = (c_min + c_max) / 2.0
    c_gauss = c_min * c_max
    # enforce that min<max
    i = np.squeeze(np.where(c_min > c_max))
    c_min[i], c_max[i] = c_max[i], c_min[i]
    u_min[i, :], u_max[i, :] = u_max[i, :], u_min[i, :]
    # flip normals to point towards vertex normals
    s = np.sign(np.sum(normals * vnormals, axis=1)).reshape(-1, 1)
    normals = normals * s
    # (here we could also project to tangent plane at vertex (using v_normals)
    # as the normals above are not really good v_normals)
    # flip u_max so that cross(u_min , u_max) aligns with normals
    u_cross = np.cross(u_min, u_max)
    d = np.sum(np.multiply(u_cross, normals), axis=1)
    i = np.squeeze(np.where(d < 0))
    u_max[i, :] = -u_max[i, :]
    return u_min, u_max, c_min, c_max, c_mean, c_gauss, normals

def smooth(surf, vfunc, n=1):
    """Smooth the mesh or a vertex function iteratively.

    Parameters
    ----------
    surf : Shape
        Mesh object
    vfunc : array
        Float vector of values at vertices, if empty, use vertex coordinates.
    n : int, default=1
        Number of iterations for smoothing.

    Returns
    -------
    vfunc : array
        Smoothed surface vertex function.
    """
    if vfunc is None:
        vfunc = surf.v
    vfunc = np.array(vfunc)
    if surf.v.shape[0] != vfunc.shape[0]:
        raise ValueError("Error: length of vfunc needs to match number of vertices")
    areas = surf.vertex_areas()[:, np.newaxis]
    adj = surf.e.copy()
    # binarize:
    adj.data = np.ones(adj.data.shape)
    # adjust columns to contain areas of vertex i
    adj2 = adj.multiply(areas)
    # rowsum is the area sum of 1-ring neighbors
    rowsum = np.sum(adj2, axis=1)
    # normalize rows to sum = 1
    adj2 = adj2.multiply(1.0 / rowsum)
    # apply sparse matrix n times (fast in spite of loop)
    vout = adj2.dot(vfunc)
    for i in range(n - 1):
        vout = adj2.dot(vout)
    return vout
    
def modularity(surf): 
    """
    Calculates the Newman spectral modularity of the mesh.

    Parameters
    ----------
    surf : Shape
        Mesh to compute Newman spectral modularity

    Returns
    -------
    float
        The Newman spectral modularity of the graph.
    """
    adj = surf.adj_sym
    
    total_degree = adj.sum()
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees_sqrt_inv = np.sqrt(1.0 / degrees)
    
    modularity_matrix = adj - degrees_sqrt_inv[:, np.newaxis] * degrees_sqrt_inv[np.newaxis, :]
                        
    eigenvalues, eigenvectors = eigs(modularity_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    eigenvectors = eigenvectors[:, sorted_indices]

    # Find the eigenvector corresponding to the largest eigenvalue
    leading_eigenvector = eigenvectors[:, 0]

    modularity = (leading_eigenvector.T @ modularity_matrix @ leading_eigenvector) / total_degree
    
    return modularity

def clustering_coefficient(surf):
    """
    Returns the clustering coefficient of the mesh.

    Parameters
    ----------
    surf : Shape
        Mesh to calculate the clustering coefficient

    Returns
    -------
    float
        Average clustering coefficient of mesh adjacency matrix

    """
    adj = surf.adj_sym
    num_verts = surf.num_verts
    # Convert the adjacency matrix to a NetworkX graph
    graph = nx.from_scipy_sparse_matrix(adj)

    return nx.average_clustering(graph)

def rich_club(surf, k=2):
    """
    Calculates the rich club coefficient of a mesh for a given degree
    threshold `k`.

    Parameters
    ----------
    surf : Shape
        Mesh to compute rich club coefficient
    k : int, optional
        Degree threshold. The default is 2.

    Returns
    -------
    float
        The rich club coefficient at degree threshold `k`

    """
    adj = surf.adj_sym
    graph = nx.from_scipy_sparse_array(adj)
    
    return nx.rich_club_coefficient(adj, normalized=True, Q=k)[0][k]

def geodesic_distmat(surf, n_jobs=1):
    """
    Compute geodesic distance using the heat diffusion method built into LaPy
        Based on: doi/10.1145/2516971.2516977
        Crane et al., "Geodesics in heat: A new approach to computing distance 
        based on heat flow"
    
    Parameters
    ----------
    surf : Shape class
        Input surface
    n_jobs : int, optional
        Number of workers to use for parallel calls to ._thread_method(),
        default is 1.
    
    Returns
    -------
    D : (N,N) np.ndarray
        Distance matrix of every vertex to every other vertex
    
    """
    if not surf.polyshape == 3:
        raise ValueError('Surface must be triangular to compute geodesic heat distance')
    
    if not surf.fem:
        surf.fem = Solver(surf, lump=True, use_cholmod=True)
        
    D = __distance_threading__(surf, surf.fem, n_jobs=n_jobs)
    
    return np.memmap(D)

def __distance_threading__(tria, fem, n_jobs=1):

    D = np.column_stack(
        Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_distance_thread_method)(tria, fem, bvert=bvert) for bvert in range(tria.v.shape[0])
            )
        )
    
    return np.asarray(D.squeeze())

def _distance_thread_method(tria, fem, bvert):    
    u = diffusion(tria, fem, vids=bvert, m=1.0)
    
    tfunc = tria_compute_gradient(tria, u)
    
    X = -tfunc / np.sqrt((tfunc**2).sum(1))[:,np.newaxis]
    X = np.nan_to_num(X)
    
    b0 = tria_compute_divergence(tria, X)
    
    chol = cholesky(fem.stiffness)
    d = chol(b0)
    
    d = d - min(d)
        
    return d

def diffusion(geometry, fem, vids, m=1.0, use_cholmod=True):
    """
    Computes heat diffusion from initial vertices in vids using
    backward Euler solution for time t [MO2]:
    
      t = m * avg_edge_length^2
    
    Parameters
    ----------
      geometry      TriaMesh or TetMesh, on which to run diffusion
      vids          vertex index or indices where initial heat is applied
      m             factor (default 1) to compute time of heat evolution:
                    t = m * avg_edge_length^2
      use_cholmod   (default True), if Cholmod is not found
                    revert to LU decomposition (slower)
    
    Returns
    -------
      vfunc         heat diffusion at vertices
    """
    sksparse = import_optional_dependency("sksparse", raise_error=use_cholmod)
    
    nv = len(geometry.v)
    fem = fem
    # time of heat evolution:
    t = m * geometry.avg_edge_length() ** 2
    # backward Euler matrix:
    hmat = fem.mass + t * fem.stiffness
    # set initial heat
    b0 = np.zeros((nv,))
    b0[np.array(vids)] = 1.0
    # solve H x = b0
    #print("Matrix Format now:  " + hmat.getformat())
    if sksparse is not None:
        print("Solver: Cholesky decomposition from scikit-sparse cholmod ...")
        chol = sksparse.cholmod.cholesky(hmat)
        vfunc = chol(b0)
    else:
        from scipy.sparse.linalg import splu
    
        print("Solver: spsolve (LU decomposition) ...")
        lu = splu(hmat)
        vfunc = lu.solve(np.float32(b0))
    return vfunc

def euclidean_distmat(surf):
    """
    Calculates the Euclidean distance matrix of each pair of vertices.

    Parameters
    ----------
    surf : Shape
        Mesh to calculate Euclidean distance matrix

    Returns
    -------
    distance_matrix : ndarray of shape=(num_verts, num_verts)
        Euclidean distance matrix

    """
    vertices = surf.v
    num_vertices = surf.num_verts
    
    distance_matrix = np.zeros((num_vertices, num_vertices))

    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            distance = np.linalg.norm(vertices[i] - vertices[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix

########################
# Mesh data operations #
########################

def parcellate_data(surf, labels):
    """
    Parcellate vertex data based on a given parcellation.

    Parameters
    ----------
    surf : Shape
        Mesh with `data` attribute of np.ndarray of shape (N, M).
    labels : np.ndarray of shape=(N,).
        label array of ints

    Returns
    -------
    ndarray
        Parcellated data of shape (num_parcels, M).
        
    Notes
    -----
    `labels` must be integer label array or file-like with integer labels for
        each vertex. Accepts ".gii", ".nii", ".vtk", ".asc", ".txt",
        and freesurfer-format label files.
        
    """
    if surf.data is None:
        raise AttributeError('Mesh must have data points to parcellate')
    
    return do.parcellate_vertex_data(surf.data, labels)

def threshold_data(surf, threshold_type='weight', threshold=0.1):
    """
    Selection of edges in mesh adjacency matrix based on `threshold_type`
    and `threshold`.

    Parameters
    ----------
    surf : Shape
        Surface with data to threshold.
    threshold_type : str, optional
        Thresholding type. The default is 'weight'.
    threshold : float, optional
        Weight to threshold based on `threshold_type`. The default is 0.1.

    Raises
    ------
    ValueError
        `threshold_type` not in `mesh.mesh_operations.threshold_types`

    Returns
    -------
    thresholded_data : csc_matrix
        Thresholded adjacency matrix
        
    Notes
    -----
    See `mesh.mesh_operations.threshold_types`

    """
    
    return do.threshold_data

def data_to_mask(surf, mask, fill=0, axis=0):
    """Assign data to mask.

    Parameters
    ----------
    surf : Shape
        Mesh with `data` array attribute
    mask : ndarray, shape = (n_mask,)
        Mask of boolean values. Data is mapped to mask.
        If `values` is 2D, the mask is applied according to `axis`.
    fill : float, optional
        Value used to fill elements outside the mask. Default is 0.
    axis : {0, 1}, optional
        If ``axis == 0`` map rows. Otherwise, map columns. Default is 0.

    Returns
    -------
    output : ndarray
        Values mapped to mask. If `values` is 1D, shape (n_mask,).
        When `values` is 2D, shape (n_samples, n_mask) if ``axis == 0`` and
        (n_mask, n_samples) otherwise.

    """

    if surf.data is None:
        raise AttributeError('Mesh must have data points to mask')
        
    return do.map_to_mask(surf.data, mask)

def optimize_recon(surf, direction='up', short=True):
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
    emodes = surf.emodes
    data = surf.data
    
    return recon.optimize_recon(surf, data, emodes, direction, short)
    
def reconstruct_data(surf, return_betas=False):
    """
    Reconstruct a dataset of `n_vertices` given a set of eigenmodes and coeffs
    conditioned on data using ordinary least squares (OLS)

    Parameters
    ----------
    surf : Shape object
        Shape object with Shape.emodes, Shape.data attributes
    return_betas : bool
        Default False
    
    Returns
    -------
    new_data : np.ndarray of (n_vertices,)
        Reconstructed data

    """
    data = surf.data
    emodes = surf.emodes
    
    return recon.reconstruct_data(surf, data, emodes, return_betas)
    
#######################################
# Triangular mesh specific operations #
#######################################

def to_tetrahedral(surf):
    """
    Get tetrahedral mesh inside bounded mesh, uses Delaunay triangulation and
    barycentric interpolation to interpolate surface data to the tetrahedral
    3D mesh.
    
    Parameters
    ----------
    surf : Shape
        Triangular mesh to transform
    
    Returns
    -------
    Shape
        Tetrahedral mesh object
    
    """
    from scipy.spatial import Delaunay
    
    # Compute Delaunay triangulation
    tri = Delaunay(surf.v)
    
    # Get the tetrahedra indices
    tetrahedra = tri.simplices
    
    # Get the vertex coordinates of the tetrahedral mesh
    tetrahedral_vertices = tri.points
    
    # Compute the faces of the tetrahedra
    tetrahedral_faces = []
    for tetra in tetrahedra:
        # Generate the four faces of the tetrahedra
        tetrahedral_faces.extend([[tetra[0], tetra[1], tetra[2]],
                      [tetra[0], tetra[1], tetra[3]],
                      [tetra[1], tetra[2], tetra[3]],
                      [tetra[0], tetra[2], tetra[3]]])

    tetrahedral_faces = np.array(tetrahedral_faces)
    
    if surf.data is not None:
        # Interpolate data from triangular surface to tetrahedra
        interpolated_data = np.zeros(len(tetrahedra))
    
        for tetra_idx, tetra in enumerate(tetrahedra):
            # Calculate the barycentric coordinates for the tetrahedron
            tetra_vertices = tetrahedral_vertices[tetra]
            barycentric_coords = tri.transform[tetra_idx, :3].dot(tetra_vertices - tri.transform[tetra_idx, 3])
    
            # Interpolate the data using the barycentric coordinates
            interpolated_value = np.sum(barycentric_coords * surf.data[tri.simplices[tetra_idx]], axis=0)
            interpolated_data[tetra_idx] = interpolated_value
        return tetrahedral_vertices, tetrahedral_faces, interpolated_data
    
    return tetrahedral_vertices, tetrahedral_faces
    
def tria_normals(surf):
    """Compute triangle normals.

    Ordering of triangles is important: counterclockwise when looking.

    Parameters
    ----------
    surf : Shape
        Triangular mesh to compute triangle normals
        
    Returns
    -------
    n : array of shape (n_triangles, 3)
        Triangle normals.
    """
    if not surf.polyshape == 3:
        raise AttributeError('Shape object must have triangular mesh')
        
    # Compute vertex coordinates and a difference vectors for each triangle:
    v0 = surf.v[surf.t[:, 0], :]
    v1 = surf.v[surf.t[:, 1], :]
    v2 = surf.v[surf.t[:, 2], :]
    v1mv0 = v1 - v0
    v2mv0 = v2 - v0
    # Compute cross product
    n = np.cross(v1mv0, v2mv0)
    ln = np.sqrt(np.sum(n * n, axis=1))
    ln[ln < sys.float_info.epsilon] = 1  # avoid division by zero
    n = n / ln.reshape(-1, 1)
    # lni = np.divide(1.0, ln)
    # n[:, 0] *= lni
    # n[:, 1] *= lni
    # n[:, 2] *= lni
    return n
        
def tria_qualities(surf):
    """
    Compute triangle quality for each triangle
    
    q = 4 sqrt(3) A / (e1^2 + e2^2 + e3^2 )
    where A is the triangle area and ei the edge length of the three edges.
    Constants are chosen so that q=1 for the equilateral triangle.

    .. note::

        This measure is used by FEMLAB and can also be found in:
        R.E. Bank, PLTMG ..., Frontiers in Appl. Math. (7), 1990.
    
    Parameters
    ----------
    surf : Shape
        Triangular mesh to compute triangle qualities
    
    Returns
    -------
    array
        Array with triangle qualities.
    """
    # Compute vertex coordinates and a difference vectors for each triangle:
    v0 = surf.v[surf.f[:, 0], :]
    v1 = surf.v[surf.f[:, 1], :]
    v2 = surf.v[surf.f[:, 2], :]
    v1mv0 = v1 - v0
    v2mv1 = v2 - v1
    v0mv2 = v0 - v2
    # Compute cross product
    n = np.cross(v1mv0, -v0mv2)
    # compute length (2*area)
    ln = np.sqrt(np.sum(n * n, axis=1))
    q = 2.0 * np.sqrt(3) * ln
    es = (v1mv0 * v1mv0).sum(1) + (v2mv1 * v2mv1).sum(1) + (v0mv2 * v0mv2).sum(1)
    return q / es
    
def tria_areas(surf):
    """Compute the area of triangles using Heron's formula.

    `Heron's formula <https://en.wikipedia.org/wiki/Heron%27s_formula>`_
    computes the area of a triangle by using the three edge lengths.
    
    Parameters
    ----------
    surf : Shape
        Triangular surface to compute triangle areas

    Returns
    -------
    areas : array
        Array with areas of each triangle.
    """
    v0 = surf.v[surf.f[:, 0], :]
    v1 = surf.v[surf.f[:, 1], :]
    v2 = surf.v[surf.f[:, 2], :]
    v1mv0 = v1 - v0
    v2mv1 = v2 - v1
    v0mv2 = v0 - v2
    a = np.sqrt(np.sum(v1mv0 * v1mv0, axis=1))
    b = np.sqrt(np.sum(v2mv1 * v2mv1, axis=1))
    c = np.sqrt(np.sum(v0mv2 * v0mv2, axis=1))
    ph = 0.5 * (a + b + c)
    areas = np.sqrt(ph * (ph - a) * (ph - b) * (ph - c))
    return areas
    
def euler(surf):
    """Compute the Euler Characteristic.

    The Euler characteristic is the number of vertices minus the number
    of edges plus the number of triangles  (= #V - #E + #T). For example,
    it is 2 for the sphere and 0 for the torus.
    This operates only on triangles array.
    
    Parameters
    ----------
    surf : Shape or TriaMesh
        Triangular surface to compute Euler characteristic.

    Returns
    -------
    int
        Euler characteristic.
    """
    # v can contain unused vertices so we get vnum from trias
    vnum = len(np.unique(surf.f.reshape(-1)))
    tnum = np.max(surf.f.shape)
    enum = int(surf.adj_sym.nnz / 2)
    return vnum - enum + tnum
    
def is_manifold(surf):
    """Check if triangle mesh is manifold (no edges with >2 triangles).

    Operates only on triangles
    
    Parameters
    ----------
    surf : Shape object with triangular mesh

    Returns
    -------
    bool
        True if no edges with > 2 triangles.
    """
    return np.max(surf.adj_sym.data) <= 2

########################################
# Tetrahedral mesh specific operations #
########################################    

def to_triangular(surf):
    """Get boundary triangle mesh of tetrahedra.

    It can have multiple connected components.
    Tria will have same vertices (including free vertices),
    so that the tria indices agree with the tet-mesh, in case we want to
    transfer information back, e.g. a FEM boundary condition, or to access
    a Shape.polyshape==3 vertex function with Shape.f indices.

    .. warning::

        Note, that it seems to be returning non-oriented triangle meshes,
        may need some debugging, until then use tria.orient_() after this.
        
    Resets any operations on surface mesh.

    Parameters
    ----------
    data : array | None
        List of tetra function values (optional).

    Returns
    -------
    Shape object
        Triangular mesh of boundary (potentially >1 components).
    triafunc : array
        List of tria function values (only returned if ``surf.data`` exists).
    """
    # get all triangles
    allt = np.vstack(
        (
            surf.f[:, np.array([3, 1, 2])],
            surf.f[:, np.array([2, 0, 3])],
            surf.f[:, np.array([1, 3, 0])],
            surf.f[:, np.array([0, 2, 1])],
        )
    )
    # sort rows so that faces are reorder in ascending order of indices
    allts = np.sort(allt, axis=1)
    # find unique trias without a neighbor
    tria, indices, count = np.unique(
        allts, axis=0, return_index=True, return_counts=True
    )
    tria = allt[indices[count == 1]]
    print("Found " + str(np.size(tria, 0)) + " triangles on boundary.")
    # if we have tetra function, map these to the boundary triangles
    if surf.data is not None:
        alltidx = np.tile(np.arange(surf.f.shape[0]), 4)
        tidx = alltidx[indices[count == 1]]
        triadata = surf.data[tidx]
        return surf.v, tria, triadata
        
    return surf.v, tria

###############################
# ShapeDNA specific functions #
###############################
    
def calc_eigs(fem, k=200):
    """Calculate the eigenvalues and eigenmodes of a surface.

    Parameters
    ----------
    fem : lapy.Solver class
        Mesh object
    num_modes : int
        Number of eigenmodes to be calculated

    Returns
    ------
    evals : array (num_modes x 1)
        Eigenvalues
    emodes : array (number of surface points x num_modes)
        Eigenmodes
    """
        
    return eigen.calc_eigs(fem, k)

