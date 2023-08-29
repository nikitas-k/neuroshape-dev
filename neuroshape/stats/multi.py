"""
Functions for performing statistical inference in multi-dimensional arrays.
Adapted from brainsmash MurrayLab 2022 by Nikitas C. Koussis 2023
"""

import numpy as np
from scipy.stats import rankdata

__all__ = ['spearmanr',
           'pearsonr',
           'pairwise_r',
           'nonparp',
           'nonparp_array',
           'zscore_array']


def spearmanr(X, Y):
    """
    Multi-dimensional Spearman rank correlation between rows of `X` and `Y`.

    Parameters
    ----------
    X : (N,P) np.ndarray
    Y : (M,P) np.ndarray

    Returns
    -------
    (N,M) np.ndarray

    Raises
    ------
    TypeError : `X` or `Y` is not array_like
    ValueError : `X` and `Y` are not same size along second axis

    """
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError('X and Y must be numpy arrays')

    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    n = X.shape[1]
    if n != Y.shape[1]:
        raise ValueError('X and Y must be same size along axis=1')

    return pearsonr(rankdata(X, axis=1), rankdata(Y, axis=1))


def pearsonr(X, Y):
    """
    Multi-dimensional Pearson correlation between rows of `X` and `Y`.

    Parameters
    ----------
    X : (N,P) np.ndarray
    Y : (M,P) np.ndarray

    Returns
    -------
    (N,M) np.ndarray

    Raises
    ------
    TypeError : `X` or `Y` is not array_like
    ValueError : `X` and `Y` are not same size along second axis

    """
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError('X and Y must be numpy arrays')

    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    n = X.shape[1]
    if n != Y.shape[1]:
        raise ValueError('X and Y must be same size along axis=1')

    mu_x = X.mean(axis=1)
    mu_y = Y.mean(axis=1)

    s_x = X.std(axis=1, ddof=n - 1)
    s_y = Y.std(axis=1, ddof=n - 1)
    cov = np.dot(X, Y.T) - n * np.dot(
        mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def pairwise_r(X, flatten=False):
    """
    Compute pairwise Pearson correlations between rows of `X`.

    Parameters
    ----------
    X : (N,M) np.ndarray
    flatten : bool, default False
        If True, return flattened upper triangular elements of corr. matrix

    Returns
    -------
    (N*(N-1)/2,) or (N,N) np.ndarray
        Pearson correlation coefficients

    """
    rp = pearsonr(X, X)
    if not flatten:
        return rp
    triu_inds = np.triu_indices_from(rp, k=1)
    return rp[triu_inds].flatten()


def nonparp(stat, dist):
    """
    Compute two-sided non-parametric p-value.

    Compute the fraction of elements in `dist` which are more extreme than
    `stat`.

    Parameters
    ----------
    stat : float
        Test statistic
    dist : (N,) np.ndarray
        Null distribution for test statistic

    Returns
    -------
    float
        Fraction of elements in `dist` which are more extreme than `stat`

    """
    n = float(len(dist))
    return np.sum(np.abs(dist) > abs(stat)) / n


def nonparp_array(X, Y):
    """
    Compute two-sided non-parametric p-value.
    
    Compute the fraction of elements in `Y` which are more extreme than
    `X`.

    Parameters
    ----------
    X : (N,) np.ndarray
        Array of real data (e.g., difference maps)
    Y : (N,n) np.ndarray
        Array of surrogate maps (e.g., null distribution of difference maps)

    Returns
    -------
    (N,) np.ndarray
        Array of fraction of elements in `dist` which are more extreme than `stat`
        
    Raises
    ------        
    TypeError : `X` or `Y` is not array_like
    ValueError : `X` and `Y` are not same size along first axis

    """
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError('X and Y must be numpy arrays')

    if X.ndim > 1:
        raise ValueError('X must have ndim=1')
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    n = X.shape[0]
    if n != Y.shape[0]:
        raise ValueError('X and Y must be same size along axis=0')
    
    n = float(len(Y))
    return np.sum(np.abs(Y) > np.abs(X), axis=0) / n


def zscore_array(X, Y):
    """
    Compute two-sided z-score.
    
    Compute the z-score of element 'i' in `X` based on the following formula:
        
               X_i - mean(Y_i)
      z_i  =  ----------------
                  std(Y_i)

    Parameters
    ----------
    X : (N,) np.ndarray
        Array of real data (e.g., difference maps)
    Y : (N,n) np.ndarray
        Array of surrogate maps (e.g., null distribution of difference maps)

    Returns
    -------
    (N,) np.ndarray
        Array of z-score elements computed by above formula

    Raises
    ------
    TypeError : `X` or `Y` is not array_like
    ValueError : `X` and `Y` are not same size along first axis
    
    """
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError('X and Y must be numpy arrays')

    if X.ndim > 1:
        raise ValueError('X must have ndim=1')
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    n = X.shape[0]
    if n != Y.shape[0]:
        raise ValueError('X and Y must be same size along axis=0')
    
    mean_Y = np.mean(Y, axis=1)
    std_Y = np.std(Y, axis=1)
    
    return (X - mean_Y) / std_Y
    