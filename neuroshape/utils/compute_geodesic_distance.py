from lapy import TriaMesh
from lapy import Solver
from lapy.DiffGeo import tria_compute_gradient, tria_compute_divergence
from sksparse.cholmod import cholesky
from neuroshape.utils.dataio import dataio
import numpy as np
from joblib import Parallel, delayed
from lapy.utils._imports import import_optional_dependency


def compute_geodesic_distance(x, n_jobs=1):
    """
    Compute geodesic distance using the heat diffusion method built into LaPy
        Based on: doi/10.1145/2516971.2516977
        Crane et al., "Geodesics in heat: A new approach to computing distance 
        based on heat flow"

    Parameters
    ----------
    x : filename or TriaMesh or np.ndarray of coords and faces
        Input filename of surface or TriaMesh of surface or tuple-like of 
        np.ndarray of coords and faces.
    n_jobs : int, optional
        Number of workers to use for parallel calls to ._thread_method(),
        default is 1.

    Returns
    -------
    D : (N,N) np.ndarray
        Distance matrix of every vertex to every other vertex

    """
    if not isinstance(x, np.ndarray): #run function
        T = dataio(x)
    
    
    tria = TriaMesh(v=T[0], t=T[1])
    fem = Solver(tria, lump=True, use_cholmod=True)
        
    D = __threading__(tria, fem, n_jobs=n_jobs)
    
    return D
    
def __threading__(tria, fem, n_jobs=1):
    
    D = np.column_stack(
        Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_thread_method)(tria, fem, bvert=bvert) for bvert in range(tria.v.shape[0])
            )
        )
    
    return np.asarray(D.squeeze())
    
def _thread_method(tria, fem, bvert):
    print(f'computing geodesic distance for vertex {bvert}')
    
    u = _diffusion(tria, fem, vids=bvert, m=1.0)
    
    tfunc = tria_compute_gradient(tria, u)
    
    X = -tfunc / np.sqrt((tfunc**2).sum(1))[:,np.newaxis]
    X = np.nan_to_num(X)
    
    b0 = tria_compute_divergence(tria, X)
    
    chol = cholesky(fem.stiffness)
    d = chol(b0)
    
    d = d - min(d)
        
    return d

def _diffusion(geometry, fem, vids, m=1.0, aniso=None, use_cholmod=True):
    """
    Computes heat diffusion from initial vertices in vids using
    backward Euler solution for time t:

      t = m * avg_edge_length^2

    :param
      geometry      TriaMesh or TetMesh, on which to run diffusion
      vids          vertex index or indices where initial heat is applied
      m             factor (default 1) to compute time of heat evolution:
                    t = m * avg_edge_length^2
      use_cholmod   (default True), if Cholmod is not found
                    revert to LU decomposition (slower)

    :return:
      vfunc         heat diffusion at vertices
    """
    sksparse = import_optional_dependency("sksparse", raise_error=use_cholmod)

    nv = len(geometry.v)
    fem = fem
    # time of heat evolution:
    t = m * geometry.avg_edge_length() ** 2
    # backward Euler matrix:
    hmat = fem.mass + t * fem.stiffness
    # set initial heat
    b0 = np.zeros((nv,))
    b0[np.array(vids)] = 1.0
    # solve H x = b0
    #print("Matrix Format now:  " + hmat.getformat())
    if sksparse is not None:
        #print("Solver: Cholesky decomposition from scikit-sparse cholmod ...")
        chol = sksparse.cholmod.cholesky(hmat)
        vfunc = chol(b0)
    else:
        from scipy.sparse.linalg import splu

        #print("Solver: spsolve (LU decomposition) ...")
        lu = splu(hmat)
        vfunc = lu.solve(np.float32(b0))
    return vfunc