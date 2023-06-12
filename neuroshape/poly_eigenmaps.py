from lapy import TriaMesh
from lapy.ShapeDNA import compute_shapedna
import numpy as np
from joblib import Parallel, delayed
from brainsmash.utils.dataio import dataio
from pathlib import Path

__all__ = ['PolyLBO']

class PolyLBO:
    """
    Description
    -----------
    Wraps the Laplace-Beltrami Operator as implemented in ShapeDNA, see:
        https://en.wikipedia.org/wiki/Laplace%E2%80%93Beltrami_operator
        and:
        http://reuter.mit.edu/software/shapedna/
        
    Runs the Laplace-Beltrami calculation in parallel (as implemented in
    .__call__() and ._call_method()) and outputs an np.ndarray of size (n,J,N) 
    of "eigenmodes" for J=200 as in Koussis et al. and where N is the number
    of vertices and n is the number of files given as input to `x`.
    
    Dependencies
    ------------
        'nibabel', 
        'lapy', 
        'nilearn', 
        'numpy', 
        'scipy', 
        'scikit-sparse',
        'neuromaps', 
        'nipype'
        
    Inputs
    ------
        See .utils.dataio()
        x : list_like or array_like or string_like or file_like : 
            list-like or array-like or string-like or file-like
            
            if array_like : ((n_vertices,3),(n_faces,3)) of np.ndarrays
            
            if string_like : string containing path of file to be loaded into
            memory. Must have .gii extension, other extensions (e.g., .vtk
            meshes) are future implementations.
            
            if file-like : string containing paths of files to be loaded into
            memory. Must have .gii extension, other extensions 
            (e.g., .vtk meshes) are future implementations.
            
        Vn : int, optional (default 2)
            Which eigenvalue to compute group contrast on. Vn cannot be greater
            than eigs.
            
        eigs : int, optional (default 2)
            Number of eigenvalues to compute using ShapeDNA.
            
        n_jobs : int, optional (default 1)
            Number of jobs to use for .compute_eigenmap()
                    
    Outputs
    -------
        See .utils.dataio()
        np.ndarray or list_like : np.ndarray or list-like
            Output type defined by input.
            
    Raises
    ------
    ValueError : input is invalid
    RuntimeError : too many files in `x`
    TypeError : x is not one of the types listed above
        
    """

    def __init__(self, eigs=200, n_jobs=1):
        self._n_jobs = n_jobs
        self._eigs = eigs + 1 #to account for the zeroth mode
        
        #self._x = [dataio(i) for i in x]
        
        #n = len(self._x)
        
        # if not n > 0:
        #     raise RuntimeError("expected filename, ndarray, string array, or .txt file, got empty")
        
    def __call__(self, x):
        """
        Generate LBO(s) for filenames in native space listed in `x`.

        Parameters
        ----------
        x : list-like or array-like or string-like or file-like

        Returns
        -------
        (n,J,N) np.ndarray
            Generated LBO eigenmap(s) of n surfaces by N = `n_vertices`

        """
        
        if type(x) == list:
            if isinstance(x[0][0], np.ndarray):
                x_ = x
            else:
                x_ = [dataio(i) for i in x]
        elif type(x) == tuple:
            try:
                return np.asarray(self.compute_eigenmaps(x)).squeeze()
            except AttributeError:
                ValueError("inputs are invalid")
        elif type(x) == str:
            x_ = dataio(x)
            return np.asarray(self.compute_eigenmaps(x=x_)).squeeze()
        elif Path(x).suffix == ".txt":
            txt = dataio(x)
            x_ = dataio(txt)
        else:
            raise TypeError("expected list-like or array-like or string-like or file-like, got {}".format(type(x)))
        
        eigmaps = Parallel(
            self._n_jobs, prefer='threads')(
                delayed(self._call_method)(x=i) for i in x_
               )
                
       
        return eigmaps
    
    def _call_method(self, x=None, hemi=None):
        """
        Subfunction used by .__call__() for parallelization purposes

        """
        #generate eigenmaps
        eigs = self._eigs
        eigvecs = self.compute_eigenmaps(x, eigs=eigs)
        
        return eigvecs.squeeze()
    
    def compute_eigenmaps(self, x, eigs):
        """
        Computes Laplace-Beltrami operator as implemented in ShapeDNA

        Parameters
        ----------
        x : (2,(N, 3)) tuple of np.ndarray
            Brain map
        eigs : int
            Number of eigenvalues to calculate on the surface `x`. 

        Returns
        -------
        (N, eigs) np.ndarray
            Eigenmaps of LBO eigenvalues of n_vertices with 
            number of columns = len(eigs)

        """
        coords, faces = x
        tria = TriaMesh(coords, faces)
        ev = compute_shapedna(tria, k=eigs)
        #eigvecs = ev['Eigenvectors'][:,Vn-1]
        #eigvecs = eigvecs/np.linalg.norm(eigvecs, ord=np.inf)
        
        
        return ev['Eigenvectors'][:,1:]
    