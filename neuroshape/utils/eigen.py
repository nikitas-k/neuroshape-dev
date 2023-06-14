import numpy as np
from neuromaps.stats import compare_images
from neuroshape.utils.swap_single_row import swap_single_row
from lapy.TriaMesh import TriaMesh
from lapy.ShapeDNA import compute_shapedna
import nibabel as nib

"""
Eigenmode helper functions (C) Systems Neuroscience Newcastle &
Nikitas C. Koussis 2023
"""

def compute_eigenmodes(surface, num_modes=200, nonzero=True):
    """
    Calculates the eigenmodes and eigenvalues of a given surface.

    Parameters
    ----------
    surface : nib.GiftiImage() or tuple of (`vertices`, `faces`)
        Surface to compute LBO
        
    num_modes : int
        Number of [0 > lambda_0 > ... > lambda_n] modes to return, default 200
        if `nonzero` is True, return only [lambda_0 > ... > lambda_n] modes.
        
    nonzero : bool
        Flag whether to return the first non-zero `num_modes` or not,
        default True

    Returns
    -------
    evals : np.ndarray of (num_modes,)
        Eigenvalues [0 > lambda_0 > ... > lambda_n] corresponding to the 
        eigenvalue solution to the Helmholtz equation solved by Finite Element 
        Method in [1]
    emodes : np.ndarray of (n_vertices, num_modes)
        Eigenmodes corresponding to the eigenfunctions solved by Finite Element
        Method in [1]
        
    References
    ----------
    [1] M. Reuter, F.-E. Wolter and N. Peinecke.
    Laplace-Beltrami spectra as "Shape-DNA" of surfaces and solids.
    Computer-Aided Design 38 (4), pp.342-366, 2006.
    http://dx.doi.org/10.1016/j.cad.2005.10.011

    """
    
    if isinstance(surface, nib.GiftiImage):
        coords, faces = (surface.darrays[0].data, surface.darrays[1].data)
        
    if type(surface) == tuple:
        coords, faces = surface
        
    if not isinstance(coords, np.ndarray) or not isinstance(faces, np.ndarray):
        raise ValueError("Input surface must be a tuple or nib.GiftiImage class with two arrays `vertices` and `faces`")
    if coords.shape[1] != faces.shape[1] and coords.shape[1] != 3:
        # try transpose?
        if coords.shape[0] == faces.shape[1]:
            coords = coords.T
        else:
            raise ValueError("Input surface has incorrect number of dimensions, must be 3-D")
            
    # make surface a TriaMesh
    tria = TriaMesh(coords, faces)
    
    # compute shapedna
    ev = compute_shapedna(tria, k=num_modes+1)
    
    if nonzero is True:
        evals = ev['Eigenvalues'][1:]
        emodes = ev['Eigenmodes'][:, 1:]
    
    else:
        evals = ev['Eigenvalues']
        emodes = ev['Eigenmodes']
    
    return evals, emodes

def maximise_recon_metric(emodes, y, metric='corr'):
    """
    Takes a set of eigenmodes `emodes` and a single eigenmode `y` and swaps 
    eigenmodes within an eigengroup to maximise `corr` in a
    reconstructed map of `y`. Other metrics are not implemented.

    Parameters
    ----------
    emodes : (N,M) np.ndarray
        Eigenmode array to swap eigenmodes within eigengroups with N 
        eigenmodes and M vertices.
    y : (M,) or (M,1) or (1,M) np.ndarray
        Functional gradient map
    metric : str, optional
        Metric to maximise. The default is 'corr'.

    Returns
    -------
    new_emodes : (N,M) np.ndarray
        Eigenmode array that maximises correlation within eigengroups.
    metric_out : float
        Maximized metric value
        
    Raises
    ------
    NotImplementedError : `metric` is not in the implemented classes

    """
        
    # swap within groups to maximise metric
    if metric != 'corr':
        raise NotImplementedError('{} is not implemented yet'.format(str(metric)))
        
    # check if y and emodes in proper orientation
    if not isinstance(emodes, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('Eigenmodes and functional maps must be array-like, got type {}, and {}'.format(type(emodes),type(y)))
    if emodes.ndim != 2 or y.ndim != 1:
        raise ValueError('Eigenmodes must be 2-D and functional map must be 1-D, got {}D and {}D'.format(emodes.ndim, y.ndim))
    if emodes.shape[1] != y.shape[0]:
        if emodes.T.shape[1] == y.shape[0]:
            emodes = emodes.T
        else:
            raise RuntimeError('Eigenmodes and functional maps must be able to be matrix multiplied, fix')
    
    groups = _get_eigengroups(emodes)
    
    # iterative swapping of eigenmodes within eigengroup to maximise metric   
    new_emodes = find_optimum_eigengroups(emodes, y, groups)
    
    return new_emodes


def _get_eigengroups(emodes):
    """
    Helper function to find eigengroups
    """
    lam = emodes.shape[1] # number of eigenmodes
    l = np.floor((lam-1)/2).astype(int)    
    if lam == 1:
        return np.asarray([0])
    if lam == 2:
        groups = [np.zeros(1).astype(int)]
        groups.append(np.ones(1).astype(int))
        return groups
    
    groups = []
    ii = 0
    i = 0
    for g in range(l+1):
        ii += 2*g+1
        if ii >= lam:
            groups.append(np.arange(i,lam-1))
            return groups
        groups.append(np.arange(i,ii))
        i = ii


def find_optimum_eigengroups(emodes, y, groups, previous_corr=0., tol=0.001):
    #copy original array and transpose for right shape
    emodes_ = emodes.T
    if len(groups) == 2:
        if len(groups[0]) < 2:
            return emodes_
    
    for group in groups:
        emodes_swapped = np.vstack(swap_single_row(emodes_[:, group]))
    
    next_betas = np.matmul(emodes_swapped.T, y)
    next_recon = np.matmul(next_betas.T, emodes_swapped).reshape(-1,)
    next_corr = compare_images(y, next_recon)
    
    if (next_corr - previous_corr) > tol:
        return emodes_.T

    emodes_ = emodes_swapped
    previous_corr = next_corr
    

def reconstruct_surface(surface, eigenmodes, n=100, normalize='area', norm_factor=1, method=None):
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
        orig_tria = TriaMesh(vertices, faces)
        new_tria = TriaMesh(new_vertices, faces)
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

    
def eigen_decomposition(data, eigenmodes, method='matrix'):
    """
    Decompose data using eigenmodes and calculate the coefficient of 
    contribution of each vector
    
    Parameters:
    -----------
    data : np.ndarray of shape (n_vertices, 3)
        N = number of vertices, P = columns of independent data
    eigenmodes : np.ndarray of shape (n_vertices, M)
        N = number of vertices, M = number of eigenmodes
    method : string
        method of calculation of coefficients: 'matrix', 'matrix_separate', 
        'regression'
    
    Returns:
    -------
    coeffs : numpy array of shape (N, 3)
     coefficient values
    
    """
    
    N, P = data.shape
    _, M = eigenmodes.shape
    
    if method == 'matrix':
        coeffs = np.linalg.solve((eigenmodes.T @ eigenmodes), (eigenmodes.T @ data))
    
    elif method == 'matrix_separate':
        coeffs = np.zeros((M, P))
        for p in range(P):
            coeffs[:, p] = np.linalg.solve((eigenmodes.T @ eigenmodes), (eigenmodes.T @ data[:, p].reshape(-1,1)))
            
    elif method == 'regression':
        coeffs = np.zeros((M, P))
        for p in range(P):
            coeffs[:, p] = np.linalg.lstsq(eigenmodes, data[:, p], rcond=None)[0]
                
    return coeffs
    
    
def compute_axes_ellipsoid(eigenvalues):
    """
    Compute the axes of an ellipsoid given the eigenmodes.
    Data is a 2D numpy array where each row is a data point and each column represents a dimension.
    """    
    return np.sqrt(eigenvalues)
    

def transform_to_spheroid(eigenvalues, eigenmodes):
    """
    Transform the eigenmodes to a spheroid space
    """
    ellipsoid_axes = compute_axes_ellipsoid(eigenvalues)
    #ellipsoid_axes = ellipsoid_axes.reshape(-1, 1)
    
    spheroid_eigenmodes = eigenmodes / ellipsoid_axes
    
    return spheroid_eigenmodes
    
    
def transform_to_ellipsoid(eigenvalues, eigenmodes):
    """
    Transform the eigenmodes in spheroid space back to ellipsoid by stretching
    """
    
    ellipsoid_axes = compute_axes_ellipsoid(eigenvalues)
    
    ellipsoid_eigenmodes = eigenmodes * ellipsoid_axes
    
    return ellipsoid_eigenmodes


def resample_spheroid(spheroid_eigenmodes, angles=None):
    """
    Resample the N-D hypersphere generated by the N orthogonal unit modes

    """
    # radius = 1 on the unit hypersphere
    r = 1
    # ensure the eigenmodes are normalized on the unit hypersphere
    spheroid_eigenmodes = spheroid_eigenmodes / np.linalg.norm(spheroid_eigenmodes, axis=0)
    
    # initialize the new points p
    p = r * spheroid_eigenmodes
    
    # check if angles are input or not
    if angles is not None:
        if angles.shape[0] != spheroid_eigenmodes.shape[1]:
            raise ValueError("The number of angles should be the same as the number of basis modes")
        angles = angles
    else:
        angles = np.random.random_sample(size=spheroid_eigenmodes.shape[1] - 1) * 2 * np.pi
        
    # Compute the coordinates for new points p
    print("Computing the coordinates for each dimension")
    for i in range(1, spheroid_eigenmodes.shape[1]):
        r_i = r
        for j in range(i):
            if np.mod(i, 2) == 1: #ODD
                if angles[j] > np.pi:
                    angles[j] = (angles[j] % np.pi)*np.pi
                r_i *= np.sin(angles[j])
            else: #EVEN
                if angles[j] > 2*np.pi: 
                    angles[j] = (angles[j] % 2*np.pi)*2*np.pi
                r_i *= np.cos(angles[j])
        if i < spheroid_eigenmodes.shape[0] - 1:
            r_i *= np.cos(angles[j-1])
        
        p += r_i * spheroid_eigenmodes[i]
        
    # find the unit modes that describe the points p
    new_modes = p / np.linalg.norm(p)
    
    # ensure that the unit modes are orthonormal (QR decomposition)
    print("Ensuring the new modes are orthonormal")
    new_modes = np.linalg.qr(new_modes, mode='reduced')[0]
    
    return new_modes  