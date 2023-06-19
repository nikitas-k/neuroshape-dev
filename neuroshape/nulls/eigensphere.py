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

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting

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

def eigenmode_resample(surface, data, evals, emodes, angles=None, decomp_method='matrix'):
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
            NOTE: for eigengroups 1, this is resampling the line,
            and for eigengroup 2, this is resampling the surface of the 
            2-sphere
        - the axes of the hyperellipse are given by the sqrt of the eigenvalues 
            corresponding to eigenvectors n
        - linear transform the eigenvectors N to the hypersphere by dividing 
            by the ellipsoid axes
        - find the set of points `p` on the hypersphere given by the basis 
            modes (the eigenmodes)
        - rotate the set of points `p` by a set given by `angles` of (0, 2*pi) for 
            even dimensions and (0, pi) for odd dimensions (resampling step)
        - find the unit vectors of the points `p` by dividing by the 
            Euclidean norm (equivalent to the eigenmodes)
        - make the new unit vectors orthonormal using QR decomposition
        - return the new eigenmodes of that eigengroup until all eigengroups
            are computed
        - reconstruct the null data by multiplying the original coefficients
            by the new eigenmodes
    
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
        
        6. <https://en.wikipedia.org/wiki/QR_decomposition>
        
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
        
    normalize : str, optional
        Normalization method for the vertices of the new surface.
        
        Accepted types:
            'constant' : normalize by a constant factor (default is 1^(1/3))
            'number' : normalize by the number of vertices
            'volume' : normalize by the volume of the reconstructed surface
            'area' : normalize by the surface area of the faces bounded by 
                    the new vertices
        
        The default is 'area'.
        
    decomp_method : str, optional
        method of calculation of coefficients: 'matrix', 'matrix_separate', 
        'regression'.
        
        The default is 'matrix'.
        
    norm_factor : int, optional
        Normalization factor for 'constant'. Unused in any other method

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
            
    # find eigengroups
    groups = _get_eigengroups(emodes)
    
    # if not given angles
    if angles is None:
        angles = np.random.random_sample(size=len(groups) - 1) * 2 * np.pi
    
    # initialize the new modes
    new_modes = np.zeros_like(emodes)
    
    # resample the hypersphere (except for groups 1 and 2)
    for group in groups:
        group_modes = emodes[:, group]
        group_evals = evals[group]
        group_new_modes = new_modes[:, group]
        
        if len(group) == 1:
            # resample along the line of real numbers (0, 1)
            group_modes *= np.random.random()
            # ensure orthonormal
            new_modes[:, group] = group_modes / np.linalg.norm(group_modes)
        
        if len(group) == 3:
            # do simple rotation
            # initialize the points
            p = group_modes
            for i in range(0, group_modes.shape[1]):
                r_i = 1 * np.sin(angles[0])
                p += r_i * group_modes[i]
            
            # ensure orthonormal
            group_new_modes = p
             # get the index for the angles
            #new_modes[:, group] = np.linalg.qr(group_new_modes, mode='reduced')[0]
        m = 1   
        # else, transform to spheroid and index the angles properly
        group_modes = transform_to_spheroid(group_evals, group_modes)
        group_new_modes = resample_spheroid(group_modes, angles[m])
        
        # transform back to ellipsoid
        new_modes[:, group] = transform_to_ellipsoid(group_evals, group_new_modes)
        m += 1
    # reconstruct the new surface
    # if normalize == 'constant':
    #     if norm_factor > 0.:
    #         new_surface = reconstruct_surface(surface, new_modes, normalize=normalize, norm_factor=norm_factor, method=method)
    #     else:
    #         raise ValueError("Normalization factor must be greater than zero")
    # else:
    #     new_surface = reconstruct_surface(surface, new_modes, n=new_modes.shape[1], normalize=normalize, method=method)
    
    coeffs = eigen_decomposition(data, emodes, method=method)
            
    surrogate_data = reconstruct_data(coeffs, new_modes, n=new_modes.shape[1])
    
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