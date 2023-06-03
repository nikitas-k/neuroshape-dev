import numpy as np
from neuromaps.stats import compare_images
from numpy.random import permutation as perm
from numpy.random import randint
from .swap_single_row import swap_single_row

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
    Helper function for .permute_within_eigengroups() to find eigengroups
    """
    lam = eigs.shape[0] # number of eigenmodes
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
        ii += 2*g + 1
        if ii >= lam:
            groups.append(np.arange(i,lam))
            break
        groups.append(np.arange(i,ii))
        i = ii
    
    return groups


def find_optimum_eigengroups(eigs, y, groups, previous_corr=0., tol=0.00001):
    #copy original array
    eigs_ = eigs
    if len(groups) == 2:
        if len(groups[0]) < 2:
            return eigs_
    
    eigs_swapped = np.vstack(swap_single_row(eigs_[groups[i]]) for i in range(len(groups)))
    next_betas = np.matmul(eigs_swapped, y)
    next_recon = np.matmul(next_betas.T, eigs_swapped).reshape(-1,)
    next_corr = compare_images(y, next_recon)
    
    try:
        if (next_corr - previous_corr) > tol:
            return eigs_
    except:
        return eigs_
    eigs_ = eigs_swapped
    previous_corr = next_corr