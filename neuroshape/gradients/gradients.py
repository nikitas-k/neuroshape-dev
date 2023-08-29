#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the connectopic Laplacian of an ROI given some fmri data
"""

import numpy as np
from scipy.sparse.csgraph import laplacian
from numpy.linalg import svd
from neuroshape.utils.normalize_data import normalize_data
import os
from sklearn.utils import check_random_state
from .mesh.data_operations import compute_similarity
from .utils import is_symmetric, make_symmetric
import warnings
from scipy.linalg import eigsh
from ..mesh.data_operations import _graph_is_connected
import scipy.sparse as ssp

from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator

os_path = dict(os.environ).get('PATH')

class Embedding(BaseEstimator, metaclass=ABCMeta):
    """Base class for embedding approaches.

    Defines fit_transform method.

    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.maps_ = None
        self.lambdas_ = None

    @abstractmethod
    def fit(self, x):
        pass

    def fit_transform(self, x):
        """Compute embedding for `x`.

        Parameters
        ----------
        x : ndarray, shape = (n, n)
            Input matrix.

        Returns
        -------
        embedding : ndarray, shape(n, n_components)
            Embedded data.

        """

        return self.fit(x).maps_

class LaplacianEigenmaps(Embedding):
    """Laplacian eigenmaps.

    Parameters
    ----------
    n_components : int or None, optional
        Number of eigenvectors. Default is 10.
    norm_laplacian : bool, optional
        If True, use normalized Laplacian. Default is True.
    random_state : int or None, optional
        Random state. Default is None.

    Attributes
    ----------
    lambdas_ : ndarray, shape (n_components,)
        Eigenvalues of the affinity matrix in ascending order.
    maps_ : ndarray, shape (n, n_components)
        Eigenvectors of the affinity matrix in same order. Where `n` is
        the number of rows of the affinity matrix.

    See Also
    --------
    :class:`.DiffusionMaps`
    :class:`.PCAMaps`

    """

    def __init__(self, n_components=10, norm_laplacian=True, random_state=None):
        super().__init__(n_components=n_components)
        self.norm_laplacian = norm_laplacian
        self.random_state = random_state

    def fit(self, affinity):
        """ Compute the Laplacian maps.

        Parameters
        ----------
        affinity : ndarray or sparse matrix, shape = (n, n)
            Affinity matrix.

        Returns
        -------
        self : object
            Returns self.

        """

        self.maps_, self.lambdas_ = laplacian_eigenmaps(affinity, 
                                n_components=self.n_components,
                                norm_laplacian=self.norm_laplacian,
                                random_state=self.random_state)

        return self
    
def laplacian_eigenmaps(adj, n_components=10, norm_laplacian=True,
                        random_state=None):
    """Compute embedding using Laplacian eigenmaps.

    Adapted from Scikit-learn to also provide eigenvalues.

    Parameters
    ----------
    adj : 2D ndarray or sparse matrix
        Affinity matrix.
    n_components : int, optional
        Number of eigenvectors. Default is 10.
    norm_laplacian : bool, optional
        If True use normalized Laplacian. Default is True.
    random_state : int or None, optional
        Random state. Default is None.

    Returns
    -------
    v : 2D ndarray, shape (n, n_components)
        Eigenvectors of the affinity matrix in same order. Where `n` is
        the number of rows of the affinity matrix.
    w : 1D ndarray, shape (n_components,)
        Eigenvalues of the affinity matrix in ascending order.

    References
    ----------
    * Belkin, M. and Niyogi, P. (2003). Laplacian Eigenmaps for
      dimensionality reduction and data representation.
      Neural Computation 15(6): 1373-96. doi:10.1162/089976603321780317

    """

    rs = check_random_state(random_state)

    # Make symmetric
    if not is_symmetric(adj, tol=1E-10):
        warnings.warn('Affinity is not symmetric. Making symmetric.')
        adj = make_symmetric(adj, check=False)

    # Check connected
    if not _graph_is_connected(adj):
        warnings.warn('Graph is not fully connected.')

    lap, dd = laplacian(adj, normed=norm_laplacian, return_diag=True)
    if norm_laplacian:
        if ssp.issparse(lap):
            lap.setdiag(1)
        else:
            np.fill_diagonal(lap, 1)

    lap *= -1
    v0 = rs.uniform(-1, 1, lap.shape[0])
    w, v = eigsh(lap, k=n_components + 1, sigma=1, which='LM', tol=0, v0=v0)

    # Sort descending and change sign of eigenvalues
    w, v = -w[::-1], v[:, ::-1]

    if norm_laplacian:
        v /= dd[:, None]

    # Drop smallest
    w, v = w[1:], v[:, 1:]

    # Consistent sign (s.t. largest value of element eigenvector is pos)
    v *= np.sign(v[np.abs(v).argmax(axis=0), range(v.shape[1])])
    return v, w

def calc_gradients(img_input, img_roi, img_mask, num_gradients=2, norm_laplacian=True,
                   random_state=True):
    
    # compute similarity matrix
    smat = compute_similarity(img_input, img_roi, img_mask)
    
    # compute laplacian eigenvectors
    egrads, evals = laplacian_eigenmaps(smat, num_components=num_gradients, 
                                        norm_laplacian=norm_laplacian,
                                        random_state=random_state)
    
    # Z-transform eigenvectors
    egrads = normalize_data(egrads)
    
    # make positive
    egrads = egrads - np.min(egrads)
    
    return evals, egrads

