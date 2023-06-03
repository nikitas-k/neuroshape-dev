import numpy as np

def check_map(x):
    """
    Check that brain map conforms to expectations.

    Parameters
    ----------
    x : np.ndarray
        Brain map

    Returns
    -------
    None

    Raises
    ------
    TypeError : `x` is not a np.ndarray object
    ValueError : `x` is not one-dimensional

    """
    if not isinstance(x, np.ndarray):
        e = "Brain map must be array-like\n"
        e += "got type {}".format(type(x))
        raise TypeError(e)
    if x.ndim != 1:
        e = "Brain map must be one-dimensional\n"
        e += "got shape {}".format(x.shape)
        raise ValueError(e)