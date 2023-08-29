import numpy as np
from ..eigen import eigen_decomposition
from ..shape import Shape

def reconstruct_surface(surface, eigenmodes, n=100, normalize='area', norm_factor=1, method='matrix'):
    """
    Reconstruct a surface of `n_vertices` given a set of eigenmodes

    Parameters
    ----------
    surface : nib.GiftiImage type
        Surface to reconstruct of `n_vertices` (must be a single hemisphere or a single surface)
    coeffs : np.ndarray of shape (n_vertices, 3)
        Coefficients of eigen decomposition
    eigenmodes : np.ndarray of shape (n_vertices, M)
        Eigenmodes of `n_vertices` by number of eigenvalues M
    n : int (default 100)
        Number of eigenmodes to use for reconstruction
    normalize : str, optional
        How to normalize the new surface. The default is 'area'.
        
        Accepted types:
            'constant' : normalize by a constant factor (default is 1^(1/3))
            'number' : normalize by the number of vertices
            'volume' : normalize by the volume of the original surface
            'area' : normalize by the area of the original vertices
            
    method : str, optional
        method of calculation of coefficients: 'matrix', 'matrix_separate', 
        'regression'
        The default is 'matrix'

    Returns
    -------
    new_surface : nib.GiftiImage type
        Reconstructed surface

    """
    
    # get existing vertices
    vertices = surface.darrays[0].data
    
    # initialize new vertices
    new_vertices = np.zeros_like(vertices)
    
    # find coeffs
    if method is not None:
        coeffs = eigen_decomposition(vertices, eigenmodes, method=method)
    else:
        coeffs = eigen_decomposition(vertices, eigenmodes)
        
    
    # partition coeffs into x, y, z
    coeffs = coeffs.T
    coeffs_x, coeffs_y, coeffs_z = coeffs
    
    new_vertices[:, 0] = eigenmodes[:, :n] @ coeffs_x
    new_vertices[:, 1] = eigenmodes[:, :n] @ coeffs_y
    new_vertices[:, 2] = eigenmodes[:, :n] @ coeffs_z

    faces = surface.darrays[1].data
    
    # normalize vertices
    if normalize:
        orig_tria = Shape(vertices, faces)
        new_tria = Shape(new_vertices, faces)
        tria_norm = new_tria
        if normalize == 'number':
            number = np.sum(orig_tria.v.shape)
            tria_norm.v = new_tria.v/(number ** (1/3))
            
        elif normalize == 'volume':
            volume = orig_tria.volume()
            tria_norm.v = new_tria.v/(volume ** (1/3))
            
        elif normalize == 'constant':
            tria_norm.v = new_tria.v/(norm_factor ** (1/3))
        
        elif normalize == 'areas':
            areas = orig_tria.vertex_areas()
            tria_norm.v = new_tria.v/(areas ** (1/3))
        
        else:
            pass
        
        new_vertices = tria_norm.v
        
    return new_vertices