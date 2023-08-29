"""
Mesh attribute functions
"""
import scipy.sparse as sparse
import numpy as np

def get_verts(surf):
    """
    Return the vertices in mesh as ndarray, shape=(num_verts,3)
    
    Parameters
    ----------
    surf : Shape
        Mesh object
        
    Returns
    -------
    ndarray of shape=(num_verts, 3)
        Vertex array
    """
    return surf.v
    
def get_faces(surf):
    """
    Return the faces in mesh as ndarray, shape=(num_faces,3)
    
    Parameters
    ----------
    surf : Shape
        Mesh object
        
    Returns
    -------
    ndarray of shape=(num_faces, 3)
        Faces array
    """
    return surf.f
    
def get_edges(surf):
    """
    Return the adjacency edge graph.
    
    Parameters
    ----------
    surf : Shape
        Mesh object
        
    Returns
    -------
    adj_sym : csc_matrix
        Adjacency matrix of vertices and edges
        
    """    
    return surf.adj_sym
    
def get_polyshape(surf):
    """
    Return whether the mesh is triangular (3) or tetrahedral (4) by taking
    the shape of the last axis.
    
    Parameters
    ----------
    surf : Shape
        Mesh object
        
    Returns
    -------
    int
        shape of last axis
        
    """
    return surf.f.shape[1]

def get_closed(surf):
    """
    Return whether the mesh is closed "True" or open "False"
    
    Parameters
    ----------
    surf : Shape
        Mesh object
        
    Returns
    -------
    bool
        "True" if mesh is closed, "False" if open
        
    """
    return False not in surf.adj_sym.data

def compute_adj_matrix(surf):
    """
    Construct symmetric adjacency matrix (edge graph) of mesh
    
    Parameters
    ----------
    surf : Shape
        Mesh object
    
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
    if surf.polyshape == 3:
        t0 = surf.f[:, 0]
        t1 = surf.f[:, 1]
        t2 = surf.f[:, 2]
        i = np.column_stack((t0, t1, t1, t2, t2, t0)).reshape(-1)
        j = np.column_stack((t1, t0, t2, t1, t0, t2)).reshape(-1)
        dat = np.ones(i.shape)
        n = surf.num_verts
        return sparse.csc_matrix((dat, (i, j)), shape=(n, n))

    else:
        t1 = surf.f[:, 0]
        t2 = surf.f[:, 1]
        t3 = surf.f[:, 2]
        t4 = surf.f[:, 3]
        i = np.column_stack((t1, t2, t2, t3, t3, 
        t1, t1, t2, t3, t4, 
        t4, t4)).reshape(-1)
        j = np.column_stack((t2, t1, t3, t2, t1, 
        t3, t4, t4, t4, t1, 
        t2, t3)).reshape(-1)
        return sparse.csc_matrix((np.ones(i.shape), (i, j)))