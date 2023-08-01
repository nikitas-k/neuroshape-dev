#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eigenmode resampling on the cortex
"""

from neuroshape.utils.eigen import (
    _get_eigengroups,
    eigen_decomposition,
    transform_to_spheroid,
    transform_to_ellipsoid,
    resample_spheroid,
    reconstruct_data,
    )

import copy

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
from brainsmash.utils.dataio import is_string_like, dataio

cmap = plt.get_cmap('viridis')

global norm_types
norm_types = [
    'constant',
    'number',
    'volume',
    'area',
    ]

global methods
methods = [
    'matrix',
    'regression',
    ]

def gram_schmidt(A):
    """Orthogonalize a set of vectors stored as the columns of matrix A."""
    # get the number of vectors
    n = A.shape[1]
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors
        for k in range(j):
            A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
        
        A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
        
    return A

def eigenmode_resample(surface, data, evals, emodes, 
                       angles=None, decomp_method='matrix', 
                       medial=None, randomize=False,
                       resample=True):
    """
    Resample the hypersphere bounded by the eigengroups contained within `emodes`,
    and reconstruct the data using coefficients conditioned on the original data
    Based on the degenerate solutions of solving the Laplace-Beltrami operator 
    on the cortex. The power spectrum is perfectly retained (the square of the 
    eigenvalues).
    
    @author: Nikitas C. Koussis, School of Psychological Sciences,
             University of Newcastle & Systems Neuroscience Group Newcastle
             
             Michael Breakspear, School of Psychological Sciences,
             University of Newcastle & Systems Neuroscience Group Newcastle
    
    Process
    -------
        - the orthonormal eigenvectors n within eigengroup l give the surface 
            of the hyperellipse of dimensions l-1
            NOTE: for eigengroup 0 with the zeroth mode, we ignore it,
            and for eigengroup 1, this is resampling the surface of the 
            2-sphere
        - the axes of the hyperellipse are given by the sqrt of the eigenvalues 
            corresponding to eigenvectors n
        - linear transform the eigenvectors N to the hypersphere by dividing 
            by the ellipsoid axes
        - finds the set of points `p` on the hypersphere given by the basis 
            modes (the eigenmodes) by normalizing them to unit length
        - rotate the set of points `p` by cos(angle) for 
            even dimensions and sin(angle) for odd dims (resampling step)
        - find the unit vectors of the points `p` by dividing by the 
            Euclidean norm (equivalent to the eigenmodes)
        - make the new unit vectors orthonormal using Gram-Schmidt process
        - return the new eigenmodes of that eigengroup until all eigengroups
            are computed
        - reconstruct the null data by multiplying the original coefficients
            by the new eigenmodes
        - resamples the null data by performing rank-ordering to replicate
            the reconstructed data term, then adds the noise term back into the
            null data to produce a surrogate that replicates the original
            variance of the empirical data.
            * Only performed if `resample` = True.
    
    References
    ----------
        1. Robinson, P. A. et al., (2016), Eigenmodes of brain activity: Neural field 
        theory predictions and comparison with experiment. NeuroImage 142, 79-98.
        <https://dx.doi.org/10.1016/j.neuroimage.2016.04.050>
        
        2. Jorgensen, M., (2014), Volumes of n-dimensional spheres and ellipsoids. 
        <https://www.whitman.edu/documents/Academics/Mathematics/2014/jorgenmd.pdf>
        
        3. <https://math.stackexchange.com/questions/3316373/ellipsoid-in-n-dimension>
        
        4. Blumenson, L. E. (1960). A derivation of n-dimensional spherical 
        coordinates. The American Mathematical Monthly, 67(1), 63-66.
        <https://www.jstor.org/stable/2308932>
        
        5. Trefethen, Lloyd N., Bau, David III, (1997). Numerical linear algebra. 
        Philadelphia, PA: Society for Industrial and Applied Mathematics. 
        ISBN 978-0-898713-61-9.
        
        6. https://en.wikipedia.org/wiki/QR_decomposition
        
        7. Chen, Y. C. et al., (2022). The individuality of shape asymmetries 
        of the human cerebral cortex. Elife, 11, e75056. 
        <https://doi.org/10.7554/eLife.75056>
        

    Parameters
    ----------
    surface : nib.GiftiImage class
        Surface to resample of surface.darrays[0].data of shape (n_vertices, 3)
        and surface.darrays[1].data of shape (n_faces, 3). Must be a single
        hemisphere or bounded surface.
        
    data : np.ndarray of shape (n_vertices,)
        Empirical data to resample
        
    evals : np.ndarray of shape (n,)
        Eigenvalues corresponding to the number of eigenmodes n in `emodes`
        
    emodes : np.ndarray of shape (n_vertices, n)
        Eigenmodes that are the solution to the generalized eigenvalue problem
        or Helmholtz equation in the Laplace-Beltrami operator of the cortex
        
    angles : np.ndarray of shape (n,), optional
        Angles to pass to eigen.resample_spheroid(). The default is None.
        If none, random angles in the half-open interval between [0, 2pi) are
        passed.
        
    decomp_method : str, optional
        method of calculation of coefficients: 'matrix', 'matrix_separate', 
        'regression'.
        
        The default is 'matrix'.
        
    medial : np.logical_array or str of path to file, default None
        Medial wall mask for the input surface `surface`. Will use the labels
        for the medial wall to mask out of the surrogates. If None, uses the
        naive implementation of finding the medial wall by finding 0.0 values
        in `data` - prone to errors if `data` has zero values outside of the
        medial wall. Can also pass `False` to not attempt masking of medial wall
        at all. 
        
        WARNING: If passing `False` to medial and `True` to resample,
        resulting surrogates may have strange distributions since the 
        rank-ordering step may assign medial-wall values outside of the 
        medial wall. USE AT YOUR OWN RISK.
        
    resample : bool, optional
        Set whether to resample surrogate map from original map to preserve
        values, default True

    Returns
    -------
    new_surface : nib.GiftiImage class
        The new surface that has been reconstructed using the new eigenmodes
        
    Raises
    ------
    ValueError : Inappropriate inputs

    """
    # perform checks
    if emodes.shape[0] != surface.darrays[0].data.shape[0]: 
        # try transpose
        emodes = emodes.T
        if emodes.shape[0] != surface.darrays[0].data.shape[0]:
            raise ValueError("Eigenmodes must have the same number of vertices as the surface")
    if evals.ndim != 1:
        raise ValueError("Eigenvalue array must be 1-dimensional")
    if emodes.shape[1] != evals.shape[0]:
        # try transpose
        emodes = emodes.T
        if emodes.shape[1] != evals.shape[0]:
            raise ValueError("There must be as many eigenmodes as there are eigenvalues")
    if not isinstance(surface, nib.GiftiImage):
        raise ValueError("Input surface must be of nibabel.GiftiImage class")
    if emodes.shape[1] >= surface.darrays[0].data.shape[0]:
        raise ValueError("Number of eigenmodes must be less than the number of vertices on the surface")
    # if normalize is not None:
    #     if normalize in norm_types:
    #         normalize = normalize
    #     else:
    #         raise ValueError("Normalization type must be 'constant', 'number', 'volume', 'area'")
    if decomp_method is not None:
        if decomp_method in methods:
            method = decomp_method
        else:
            raise ValueError("Eigenmode decomposition method must be 'matrix' or 'regression'")
    
    # mask out medial wall
    if medial is None:
        # get index of medial wall hopefully
        medial_wall = np.abs(data) == 0.0
        
    elif medial is False:
        if resample is True:
            raise RuntimeWarning("Resampling without masking out the medial wall "
                                 "may result in erroneous surrogate distributions. "
                                 "The authors of this code do not take responsibility for "
                                 "improper usage.\n"
                                 "USE AT YOUR OWN RISK.")
            
            medial_wall = np.zeros_like(data)
        
    else: # if given medial array
        if is_string_like(medial) == True:
            try:
                medial = dataio(medial)
            except:
                raise RuntimeError("Could not load medial wall file, please check")
        
        if isinstance(medial, np.ndarray) == True:
            if medial.ndim != 1:
                raise ValueError("Medial wall array must be a vector")
            if medial.shape[0] != surface.darrays[0].data.shape[0]:
                # try transpose
                if medial.shape[1] != surface.darrays[0].data.shape[0]:
                    raise ValueError("Medial wall array must have the same number of vertices as the brain map")
                else:
                    medial_wall = medial.T
            if not np.array_equal(medial, medial.astype(bool)):
                raise RuntimeError("Medial wall array must be 1 for the ROI (medial wall) and 0 elsewhere")
            else:    
                medial_wall = medial
        else:
            raise ValueError("Could not use provided medial wall array or "
                             "file, please check")
    
    medial_wall = medial_wall.astype(bool)
    
    medial_mask = np.logical_not(medial_wall)
    data_copy = copy.deepcopy(data) # deepcopy original data so it's not modified
    data_copy = data_copy[medial_mask]
    emodes_copy = copy.deepcopy(emodes)
    emodes_copy = emodes_copy[medial_mask]
    #emodes_copy = gram_schmidt(emodes_copy)
    
    # find eigengroups
    groups = _get_eigengroups(emodes_copy)
    
    # if not given angles
    if angles is None:
        angles = np.random.randn(len(groups) + 1) * np.pi
    
    # initialize the new modes
    new_modes = copy.deepcopy(np.zeros_like(emodes_copy))
    
    m = 0 #index of angles
    # resample the hypersphere (except for groups 1 and 2)
    for idx in range(1, len(groups)):
        group_modes = emodes_copy[:, groups[idx]]
        group_evals = evals[groups[idx]]
        
        if len(groups[idx]) == 3:
            # do simple rotation
            # initialize thße points
            p = group_modes / np.sqrt(group_evals)
            #p /= np.linalg.norm(p, axis=0)
            p *= np.cos(angles[m])
            
            # ensure orthonormal
            group_new_modes = gram_schmidt(p * np.sqrt(group_evals))
             # get the index for the angles
            new_modes[:, groups[idx]] = group_new_modes
        
        m += 1
        # else, transform to spheroid and index the angles properly
        group_new_modes = transform_to_spheroid(group_evals, group_modes)
        group_spherical_modes = resample_spheroid(group_new_modes, angles[m])
        
        # ensure orthonormal
        #group_spherical_modes = gram_schmidt(group_spherical_modes)
        
        # transform back to ellipsoid
        group_ellipsoid_modes = transform_to_ellipsoid(group_evals, group_spherical_modes)
        
        new_modes[:, groups[idx]] = gram_schmidt(group_ellipsoid_modes) #/ np.linalg.norm(group_ellipsoid_modes)
     
    #new_modes = gram_schmidt(new_modes) #/ np.linalg.norm(new_modes)
    # find the coefficents of the modes to the data by solving the OLS
    # decomposition, either through the normal equation solution or regression    
    coeffs = eigen_decomposition(data_copy, emodes_copy, method=method)
    reconstructed_data = coeffs @ emodes_copy.T
    
    # matrix multiply the estimated coefficients by the new modes
    surrogate_data = np.zeros_like(data)
    
    if randomize is True:
        for i in range(len(groups)):
            coeffs[groups[i]] = np.random.permutation(coeffs[groups[i]])
    
    surrogate_data[medial_mask] = reconstruct_data(coeffs, new_modes)
    
    # mask out medial wall from surrogate
    #surrogate_data[medial_wall] = 0.0
    mask = np.logical_not(medial_wall)

    # Mask the data and surrogate_data excluding the medial wall
    surr_no_mwall = copy.deepcopy(surrogate_data)
    surr_no_mwall = surr_no_mwall[mask]
    
    if resample is True:
        # Get the rank ordered indices
        data_ranks = reconstructed_data.argsort()[::-1]
        surr_ranks = surr_no_mwall.argsort()[::-1]
        
        # Resample surr_no_mwall according to the rank ordering of data_no_mwall
        surr_no_mwall[surr_ranks] = reconstructed_data[data_ranks]
        
    else: # force match the minima
        surr_no_mwall /= np.sqrt(np.mean(surr_no_mwall**2))
        surr_no_mwall *= np.sqrt(np.mean(data_copy**2))        
    
    # now add the residuals of the original data
    residuals = data_copy - reconstructed_data

    # slightly permute the original residuals
    #window_size = 50  # size of the local section to shuffle
    
    # Loop over the array with a stride of window_size
    #for i in range(0, len(residuals), window_size):
        # Randomly permute a small section of the array
        #residuals[i:i+window_size] = np.random.permutation(residuals[i:i+window_size])
    
    surr_no_mwall = surr_no_mwall + residuals
    
    output_surr = np.zeros_like(surrogate_data)
    output_surr[mask] = surr_no_mwall
    
    surrogate_data = output_surr
    
    return surrogate_data


def plot_data(surface, data, hemi='left', view='lateral', cmap='gray', show=True):
    """
    Plots a data map using nilearn.plotting, returns fig and ax handles
    from matplotlib.pyplot for further use. Can also plot values on the
    surface by input to `data`.

    Parameters
    ----------
    surface : nib.GiftiImage class or np.ndarray of shape (n_vertices, 3)
        A single surface to plot.
    data : np.ndarray of shape (n_vertices,)
        Data to plot on the surface
    hemi : str, optional
        Which hemisphere to plot. The default is 'left'.
    view : str, optional
        Which view to look at the surface. 
        The default is 'lateral'. Accepted strings are detailed in
        the docs for nilearn.plotting
    cmap : str or matplotlib.cm class
        Which colormap to plot the surface with, default is 'viridis'
    show : bool, optional
        Flag whether to show the plot, default is True

    Returns
    -------
    fig : figure handle    
    ax : axes handle

    """
    # make figure
    fig = plt.figure(figsize=(15,9), constrained_layout=False)
    mesh = (surface.darrays[0].data, surface.darrays[1].data)
    
    # get colormap
    cmap = plt.get_cmap(cmap)
    vmin = np.min(data)
    vmax = np.max(data)
        
    # plot surface
    ax = fig.add_subplot(projection='3d')
    plotting.plot_surf(mesh, surf_map=data, hemi=hemi, view=view, 
                       vmin=vmin, vmax=vmax, colorbar=False, 
                       cmap=cmap, axes=ax)
    ax.dist = 7
    
    # show figure check
    if show is True:
        plt.show()
    
    return fig, ax
    

def eigen_spectrum(evals, show=True):
    """
    Plot the eigenvalue power spectrum. Returns fig and ax handles from
    matplotlib.pyplot for further use.

    Parameters
    ----------
    evals : np.ndarray
        Number of eigenvalues
    show : bool, optional
        Flag whether to show plot, default True

    Returns
    -------
    fig : matplotlib.pyplot class
        Figure handle
    ax : matplotlib.pyplot class
        Axes handle

    """
    
    # compute power spectrum = eval^2
    
    power = evals*evals
    
    # now do figure
    fig = plt.figure(figsize=(15, 9), constrained_layout=False)
    
    ax = fig.add_subplot()
    
    markerline, _, _ = plt.stem(np.arange(1,len(evals)+1), power, linefmt='blue', markerfmt=None)
    
    plt.xlabel(r'Eigenvalue $\lambda$')
    plt.ylabel(r'Eigenvalue power $\lambda^2$')
    
    # show figure check
    if show is True:
        plt.show()
    
    return fig, ax