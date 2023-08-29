import numpy as np
from joblib import Parallel, delayed
from numpy.random import Generator, PCG64

class Permutation:
    """
    Parallel implementation of permutation of `np.ndarray`, where the objective 
    is to calculate a null distribution of group average metric files
    for significance testing.
            
    Parameters
    ----------
    x : (n,N) np.ndarray
        Array of n subjects by N = `n_vertices`. NOTE: Must be in common space
        {'fsLR', 'fsaverage'}. See neuromaps.datasets.fetch_atlas()
        
    y : (m,N) np.ndarray
        Array of m subjects by N = `n_vertices`. NOTE: Must be in common space
        {'fsLR', 'fsaverage'}. See neuromaps.datasets.fetch_atlas()
        
        NOTE
        ----
        n does not have to equal m but N must be the same.
    
    n_perm : int
        Number of permutations
        
    seed : None or int
        Specify the seed for random number generation
        
    n_jobs : int
        Number of jobs to use for parallelization
        
    Returns
    -------
    (n_perm,n,N) np.ndarray
        All surrogates for group x
    
    (n_perm,m,N) np.ndarray
        All surrogates for group y
    
    Raises
    ------
    ValueError: `x` or `y` is not np.ndarray
    RuntimeError: `x` and `y` do not have the same number of vertices
    
    """
    def __init__(self, x, y, n_perm=100, seed=None, n_jobs=1):
        
        self._n_perm = n_perm
        self._seed = seed
        self._n_jobs = n_jobs
        
        if type(x) is not np.ndarray or type(y) is not np.ndarray:
            raise ValueError("Input is not `np.ndarray`, is {} instead".format(type(x)))
        if x.shape[1] != y.shape[1]:
            raise RuntimeError("Inputs `x` and `y` must have the same number of vertices")
        
        self._x = x
        self._y = y
        
        if seed == None:
            self._rng = np.random.default_rng()
        else:
            self._rng = Generator(PCG64(seed))
        
    def __call__(self, n_perm):
        """
        Generate permuted arrays for arrays given by `x` and `y`

        Parameters
        ----------
        n_perm : number of permuted arrays to return

        Returns
        -------
        (n,N) np.ndarray
            Shuffled group x
        (n,N) np.ndarray
            Shuffed group y

        """
        rs = np.arange(n_perm)
        surr_x, surr_y = np.stack(
            Parallel(
                self._n_jobs, prefer='threads')(
                delayed(self._call_method)(self._x, self._y, self._rng, rs=i) for i in rs
            ), axis=1
        )
        return [np.asarray([*surr_x]).squeeze(), 
                np.asarray([*surr_y]).squeeze()
            ]
                        
        
    def _call_method(self, x, y, rng, rs):
        """
        Subfunction used by .permutation() for parallelization purposes
        """    
        #generate new arrays        
        surr_x, surr_y = self.permute_array(x, y, rng)
        
        return surr_x.squeeze(), surr_y.squeeze()
            
    
    def permute_array(self, x, y, rng):
        """
        Generates permuted subject lists from `x` to generate null distribution.
    
        Parameters
        ----------
        x : (n,N) np.ndarray
        
        y : (m,N) np.ndarray
        
            NOTE
            ----
            n does not have to equal m but N must be the same.
            
        rng : None or np.random.Generator method
            Set for reproducibility, default None
    
        Returns
        -------
        surr_x : (n,N) np.ndarray
            Surrogate array from reshuffling random m subjects into n
            
        surr_y : (m,N) np.ndarray
            Surrogate array from reshuffling random n subjects into m
    
        """
        n = x.shape[0]
        #m = y.shape[0]
        
        #pool groups to shuffle
        data = np.vstack((x, y))
        #shuffle
        rng.shuffle(data, axis=0)
        
        surr_x, surr_y = data[:n], data[n+1:]
        
        return surr_x, surr_y
    
    