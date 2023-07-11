from scipy.interpolate import griddata
import numpy as np
from ..utils.concavehull import ConcaveHull
from shapely.geometry import Point

def grid(x, vertices, tol=20, spacing=1.0, ch=None):
    """
    Interpolates surface onto grid
    
    Parameters
    ----------
    x : (N,) np.ndarray
        Data values at each surface vertex
    vertices : (N,3) np.ndarray
        Vertices of gifti object
    tol : float
        Tolerance for calculation of concave hull, default 20
    spacing : float
        Target grid spacing in mm, default 1.0
    
    Returns
    -------
    interpolated_data : (N,M) np.ndarray
        Interpolated data on the grid 'xi'
    gridX : (N,) np.ndarray
        Interpolated x coordinates of vertices on the grid
    gridY : (N,) np.ndarray
        Interpolated y coordinates of vertices on the grid   
    
    Notes
    -----
    Data values outside of convex hull are left as 'np.nan'
    """
    
    c = x
    x = vertices[:, 0]
    y = vertices[:, 1]
    gridX, gridY = np.mgrid[min(x):max(x):spacing, min(y):max(y):spacing]
    
    unbounded_data = griddata((x, y), c, (gridX, gridY), method='nearest', fill_value=0)
    
    ch = ConcaveHull()
    ch.loadpoints(vertices[:, :2])
    ch.calculatehull(tol=tol)
    
    bounded_data = np.zeros(unbounded_data.shape)*np.nan
    for i in np.arange(len(unbounded_data)):
        for j in np.arange(len(unbounded_data[i])):
            if ch.boundary.contains(Point(gridX[i,j], gridY[i,j])):
                bounded_data[i,j] = unbounded_data[i,j]
            else:
                continue
            
    bounded_data[np.isnan(bounded_data)] = 0
        
    return bounded_data, gridX, gridY

def mesh(interpolated_data, vertices, gridX, gridY):
    """
    Backprojects grid from 'grid()' to surface
    
    Parameters
    ----------
    interpolated_data : (N,M) np.ndarray
        Output from 'grid()', interpolated grid
    vertices : (N,3) np.ndarray
        Vertices of gifti object
    gridX : (N,N) np.ndarray
        Interpolated X coordinates
    gridY : (N,N) np.ndarray
        Interpolated Y coordinates
    
    Returns
    -------
    uninterpolated_data : (N,)
        Uninterpolated data output from backprojecting grid onto vertex coordinates
    
    Notes
    -----
    Data values outside of convex hull are left as 'np.nan'
    """
    c = np.matrix.flatten(interpolated_data)    
    x = vertices[:, 0]
    y = vertices[:, 1]
    uninterpolated_data = griddata((gridX.flatten(), gridY.flatten()), c, (x, y), method='linear')
    
    
    return uninterpolated_data
