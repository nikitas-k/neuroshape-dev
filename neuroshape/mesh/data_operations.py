"""
Data operation functions
"""
import numpy as np
from . import iosupport as io
#from ..kernels import check_kernels, gaussian, exp, log
from scipy.sparse.csgraph import connected_components
from skimage.filters import gaussian

def _graph_is_connected(graph):
    return connected_components(graph)[0] == 1

def smooth_data(obj, fwhm, kernel='gaussian', axis=-1):
    """
    Smooth input data with different kernels and 
    sigma = fwhm / sqrt(8 * log(2)).

    Parameters
    ----------
    obj : np.ndarray or filename
        Input data of any shape, by default smooths along last axis (-1)
    fwhm : float
        Full-width half-maximum for kernel to smooth input.
    kernel : str
        Smoothing kernel
            Types:
                'gaussian', 'exp', 'log'

    Returns
    -------
    ndarray, shape=`obj.shape`
        Smoothed data

    """
    # load data
    data = io.read(obj)
    
    # last axis
    T = data.shape[axis]
    
    # sigma
    sigma = fwhm / np.sqrt(8 * np.log(2))
    
    # initialize smoothed image
    smoothed_data = np.zeros_like(data)
    
    # if kernel == 'gaussian':
    #     kernel = gaussian
    
    for vol in range(T):
        smoothed_data[::vol] = gaussian(data[::vol], sigma=sigma)
        
    return smoothed_data

def normalize_data(data):
    data_normalized = np.subtract(data, np.mean(data, axis=0))
    data_normalized = np.divide(data_normalized, np.std(data_normalized, axis=0))
    
    return data_normalized

def swap_single_row(x):
    new_x = x
    idx = np.random.randint(0, len(x), size=2)
    new_idx = idx[::-1]
    new_x[new_idx] = x[idx]
    return new_x

# def compute_similarity(obj, mask):
    
    
#     data = normalize_data(data)
#     T = data.shape[0]
    
#     data_msk = 
#     U, S, _ = svd(data, full_matrices=False)
#     a = U.dot(np.diag(S))
#     a = a[:, :-1]
    
#     a = normalize_data(a)
    
#     C = 
    
def parcellate_vertex_data(data, labels):
    """
    Parcellate vertex data based on a given parcellation.

    Parameters
    ----------
    labels : str of filename or np.ndarray of shape=(N,).
        label array of ints
    data : np.ndarray of shape (N, M).
        Data to parcellate of shape (N, M).

    Returns
    -------
    ndarray
        Parcellated data of shape (num_parcels, M).
        
    Notes
    -----
    `labels` must be integer label array or file-like with integer labels for
        each vertex.
    """
    if io.is_string_like(labels):
        try:
            labels = io.load(labels)
        except Exception as e:
            raise RuntimeError('{e}')        
    
    num_vertices = labels.shape[0]
    parcels = np.unique(labels[labels > 0])
    num_parcels = len(parcels)

    if data.shape[0] != num_vertices:
        data = data.T
        if data.shape[0] != num_vertices:
            raise ValueError('Number of data points must be equal to number of vertices')

    data_parcellated = np.zeros((num_parcels, data.shape[1]))

    for parcel_ind in range(num_parcels):
        parcel_interest = parcels[parcel_ind]

        ind_parcel = np.where(labels == parcel_interest)[0]

        data_parcellated[parcel_ind, :] = np.nanmean(data[ind_parcel, :], axis=0)

    return data_parcellated

def map_to_mask(data, mask, fill=0, axis=0):
    """Assign data to mask.

    Parameters
    ----------
    data : ndarray, shape = (n_values,) or (n_samples, n_values)
        Source array of values.
    mask : ndarray, shape = (n_mask,)
        Mask of boolean values. Data is mapped to mask.
        If `values` is 2D, the mask is applied according to `axis`.
    fill : float, optional
        Value used to fill elements outside the mask. Default is 0.
    axis : {0, 1}, optional
        If ``axis == 0`` map rows. Otherwise, map columns. Default is 0.

    Returns
    -------
    output : ndarray
        Values mapped to mask. If `values` is 1D, shape (n_mask,).
        When `values` is 2D, shape (n_samples, n_mask) if ``axis == 0`` and
        (n_mask, n_samples) otherwise.

    """

    if np.issubdtype(data.dtype, np.integer) and not np.isfinite(fill):
        raise ValueError("Cannot use non-finite 'fill' with integer arrays.")

    if axis == 1 and data.ndim > 1:
        data = data.T

    values2d = np.atleast_2d(data)
    if values2d.shape[1] > np.count_nonzero(mask):
        raise ValueError("Mask cannot allocate values.")

    mapped = np.full((values2d.shape[0], mask.size), fill, dtype=data.dtype)
    mapped[:, mask] = values2d
    if data.ndim == 1:
        return mapped[0]
    if axis == 1:
        return mapped.T
    return mapped