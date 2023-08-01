import numpy as np
from neuromaps.stats import compare_images
from neuroshape.utils.swap_single_row import swap_single_row
from lapy.TriaMesh import TriaMesh
from scipy.special import sph_harm

"""
Eigenmode helper functions (C) Systems Neuroscience Newcastle &
Nikitas C. Koussis 2023
"""

def maximise_recon_metric(eigs, y, metric='corr'):
    """
    Takes a set of eigenmodes `eigs` and a single eigenmode `y` and swaps 
    eigenmodes within an eigengroup to maximise `corr` in a
    reconstructed map of `y`. Other metrics are not implemented.

    Parameters
    ----------
    eigs : (N,M) np.ndarray
        Eigenmode array to swap eigenmodes within eigengroups with N 
        eigenmodes and M vertices.
    y : (M,) or (M,1) or (1,M) np.ndarray
        Functional gradient map
    metric : str, optional
        Metric to maximise. The default is 'corr'.

    Returns
    -------
    new_eigs : (N,M) np.ndarray
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
        
    # check if y and eigs in proper orientation
    if not isinstance(eigs, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('Eigenmodes and functional maps must be array-like, got type {}, and {}'.format(type(eigs),type(y)))
    if eigs.ndim != 2 or y.ndim != 1:
        raise ValueError('Eigenmodes must be 2-D and functional map must be 1-D, got {}D and {}D'.format(eigs.ndim, y.ndim))
    if eigs.shape[1] != y.shape[0]:
        if eigs.T.shape[1] == y.shape[0]:
            eigs = eigs.T
        else:
            raise RuntimeError('Eigenmodes and functional maps must be able to be matrix multiplied, fix')
    
    groups = _get_eigengroups(eigs)
    
    # iterative swapping of eigenmodes within eigengroup to maximise metric   
    new_eigs = find_optimum_eigengroups(eigs, y, groups)
    
    return new_eigs


def _get_eigengroups(eigs):
    """
    Helper function to find eigengroups
    """
    print("""IMPORTANT: EIGENMODES MUST NOT BE TRUNCATED AT FIRST NON-ZERO MODE
    FOR THIS FUNCTION TO WORK""")
    lam = eigs.shape[1] # number of eigenmodes
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
            groups.append(np.arange(i,lam))
            return groups
        groups.append(np.arange(i,ii))
        i = ii


def find_optimum_eigengroups(eigs, y, groups, previous_corr=0., tol=0.001):
    #copy original array and transpose for right shape
    eigs_ = eigs.T
    if len(groups) == 2:
        if len(groups[0]) < 2:
            return eigs_
    
    eigs_swapped = np.vstack(swap_single_row(eigs_[:, groups[i]]) for i in range(len(groups)))
    next_betas = np.matmul(eigs_swapped, y)
    next_recon = np.matmul(next_betas.T, eigs_swapped).reshape(-1,)
    next_corr = compare_images(y, next_recon)
    
    try:
        if (next_corr - previous_corr) > tol:
            return eigs_.T
    except:
        return eigs_.T
    eigs_ = eigs_swapped
    previous_corr = next_corr
    

def reconstruct_data(coeffs, eigenmodes):
    """
    Reconstruct a dataset of `n_vertices` given a set of eigenmodes and coeffs
    conditioned on data using ordinary least squares (OLS)

    Parameters
    ----------
    coeffs : np.ndarray of shape (M,)
        Coefficients output from fitting OLS
    eigenmodes : np.ndarray of shape (n_vertices, M)
        Eigenmodes of `n_vertices` by number of eigenvalues M
    pv : float (default 100), rounds up to nearest index
        Percentage of `eigenmodes` to use for reconstruction (default 100%)

    Returns
    -------
    new_data : np.ndarray of (n_vertices,)
        Reconstructed data

    """
    
    coeffs = coeffs.reshape(-1, 1)
    
    new_data = eigenmodes @ coeffs
    
    return new_data.squeeze()
    

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
    
    if data.ndim > 1:
        N, P = data.shape
    else:
        P = 1
    
    _, M = eigenmodes.shape
    
    if method == 'matrix':
        print("Using matrix decomposition to reconstruct data")
        coeffs = np.linalg.solve((eigenmodes.T @ eigenmodes), (eigenmodes.T @ data))
            
    elif method == 'regression':
        print("Using regression method to reconstruct data")
        coeffs = np.zeros((M, P))
        if P > 1:
            for p in range(P):
                coeffs[:, p] = np.linalg.lstsq(eigenmodes, data[:, p], rcond=None)[0]
        else:
            coeffs = np.linalg.lstsq(eigenmodes, data, rcond=None)[0]
            
    else:
        raise ValueError("Accepted methods for decomposition are 'matrix', and 'regression'")
                
    return coeffs
    
    
def compute_axes_ellipsoid(eigenvalues):
    """
    Compute the axes of an ellipsoid given the eigenmodes.
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


def resample_spheroid(spheroid_eigenmodes, angle):
    """
    Resample the N-D hypersphere generated by the N orthogonal unit modes

    """
    # ensure the eigenmodes are normalized on the unit hypersphere
    spheroid_eigenmodes = spheroid_eigenmodes #/ np.linalg.norm(spheroid_eigenmodes, axis=0)
    
    # length of group to determine evenness or oddness of dimensions
    #dims = spheroid_eigenmodes.shape[0]
    
    # initialize the new points p
    p = spheroid_eigenmodes * np.cos(angle)
    
    # compute the coordinates for the new points
    # for i in range(0, spheroid_eigenmodes.shape[1]):
    #     p[:, i] *= np.cos(angle)
    
    # Compute the coordinates for new points p
    # print("Computing the coordinates for each dimension, multiplying by single angle per group")
    # if dims % 2 == 0: # EVEN
    #     p *= np.cos(angle)
    # else: #dims % 2 == 1 ODD
    #     p *= np.sin(angle)
    
    # ensure that the unit modes are orthonormal (QR decomposition)
    new_modes = p
    
    return new_modes

def _cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.
    
    Parameters:
    x, y, z: Cartesian coordinates
    
    Returns:
    r: radius
    theta: azimuthal angle in radians
    phi: polar angle in radians
    """
    r = np.sqrt(x**2 + y**2 + z**2)  # radius
    theta = np.arctan2(y, x)  # azimuthal angle
    phi = np.arccos(z / r)  # polar angle

    # Adjust theta to be in range [0, 2pi)
    theta = theta if theta >= 0 else theta + 2 * np.pi

    return r, theta, phi

def compute_spherical_harmonics(l, m, theta, phi):
    """
    Function to compute the spherical harmonics Y_{lm}(theta, phi)
    
    Parameters:
    l: int - The degree of the spherical harmonic, should be >= 0.
    m: int - The order of the spherical harmonic, should be -l <= m <= l.
    theta: float - The azimuthal (longitudinal) coordinate, 0 <= theta < 2*pi.
    phi: float - The polar (colatitudinal) coordinate, 0 <= phi <= pi.

    Returns:
    The value of the spherical harmonic Y_{lm}(theta, phi)
    """
    # Compute the spherical harmonics
    Y_lm = sph_harm(m, l, theta, phi)
    return Y_lm