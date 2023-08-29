import numpy as np
from ..eigen import _get_eigengroups, eigen_decomposition
from ..mesh.data_operations import swap_single_row
from ..stats.brainstats import compare_images

def optimize_recon(emodes, data, direction='up', short=True):
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
    
    #TODO
    
    return reconstructed_corr

def maximise_recon_metric(eigs, y):
    """
    Takes a set of eigenmodes `eigs` and a single data array `y` and swaps 
    eigenmodes within an eigengroup to maximise `corr` (pearson correlation)
    in a reconstructed map of `y`.

    Parameters
    ----------
    eigs : (N,M) np.ndarray
        Eigenmode array to swap eigenmodes within eigengroups with N 
        eigenmodes and M vertices.
    y : (M,) or (M,1) or (1,M) np.ndarray
        Functional gradient map

    Returns
    -------
    new_eigs : (N,M) np.ndarray
        Eigenmode array that maximises correlation with reconstruction.
    metric_out : float
        Maximized metric value

    """
          
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
    

def reconstruct_data(eigenmodes, data):
    """
    Reconstruct a dataset of `n_vertices` given a set of eigenmodes and coeffs
    conditioned on data using ordinary least squares (OLS)

    Parameters
    ----------
    eigenmodes : np.ndarray of shape (n_vertices, M)
        Eigenmodes of `n_vertices` by number of eigenvalues M
    data : np.ndarray of shape (n_vertices,)
        Data to reconstruct

    Returns
    -------
    new_data : np.ndarray of (n_vertices,)
        Reconstructed data

    """
    coeffs = eigen_decomposition(data, eigenmodes)
    
    coeffs = coeffs.reshape(-1, 1)
    new_data = coeffs.T @ eigenmodes.T
    
    return new_data.squeeze()