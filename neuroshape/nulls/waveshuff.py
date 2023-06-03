"""
Creates surrogate data by permuting wavelet coefficients for 2-D data
Only shuffles wavelet coefficients where the data is non-zero
Based on M. Breakspear 2002,2022
Updated for Python by N. Koussis 2022
Based on Sampled from Murray lab brainsmash 2022
"""

from ..utils.dataio import dataio
from ..utils.concavehull import ConcaveHull
from ..utils.load_mesh_vertices import load_mesh_vertices
from ..utils.checks import check_map, check_pv, check_deltas
from ..utils.meshtransforms import grid, mesh
from ..utils.shuffle_matrix import shuffle_matrix
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_random_state
import numpy as np
from joblib import Parallel, delayed
import pywt
import nibabel as nib
from scipy.interpolate import griddata
from shapely.geometry import Point

__all__ = ['Waveshuff']

class Waveshuff:
    """
    Sampling implementation of waveshuffling generator
    
    Parameters
    ----------
    x : filename or 1D np.ndarray
        Target brain map
    D : filename or (N,N) np.ndarray or np.memmap
        Pairwise distance matrix between elements of 'x'. Each row of 'D' should
        be sorted. Indices used to sort each row are passed to the 'index' argument.
        index : filename or (N,N) np.ndarray or np.memmap
    vx : filename 
        Path to flatmap gifti file
    ns : int, default 500
        Take a subsample of 'ns' rows from 'D' when fitting variograms
    deltas : np.ndarray or List[float], default [0.3, 0.5, 0.7, 0.9]
        Proportions of neighbors to include for smoothing, in (0, 1)
    pv : int, default 70
        Percentile of the pairwise distance distribution (in 'D') at which to truncate
        during variogram fitting
    nh : int, default 25
        Number of uniformly space distances at which to compute variogram
    knn : int, default 1000
        Number of nearest regions to keep in the neighborhood of each region
    b : float or None, default None
        Gaussian kernel bandwidth for variogram smoothing. If None,
        three times the distance interval spacing is used.
    scales : int or (N,) np.ndarray
        Scale on which to perform wavelet transform (default 1:8)
    wv : string, default 'db8'
        Wavelet transform, accepts ...
    B : float, default 0
        Size of boundary to resample separately
    seed : None or int or np.random.RandomState instance (default 1)
        Specify the seed for random number generation (or random state instance)
    n_jobs : int (default 1)
        Number of jobs to use for parallelizing creation of surrogate maps
    waveamp : int or None, default 1
        Do wavelet amplitude-adjusted step
    """

    def __init__(self, x, D, index, vx=None, ns=500, pv=70, nh=25, knn=1000, b=None,
                 deltas=np.arange(0.3, 1., 0.2), scales=np.arange(8), wv='db8',
                 B=0, seed=1, n_jobs=1, ampadj=True, ch=None):

        self._rs = check_random_state(seed)
        self._n_jobs = n_jobs

        self.x = x
        n = self._x.size
        self.nmap = int(n)
        self.knn = knn
        self.D = D
        self.index = index
        self.vx = vx
        self.nh = int(nh)
        self.deltas = deltas
        self.ns = int(ns)
        self.b = b
        self.pv = pv
        self._ikn = np.arange(self._nmap)[:, None]
        self.ch = ch
        
        self._dmax = np.percentile(self._D, self._pv)
        self.h = np.linspace(self._D.min(), self._dmax, self._nh)
        self.scales = scales
        self.wv = wv
        self.B = B
        self.ampadj = True
        
        if vx:
            self.vertices = load_mesh_vertices(vx)
        else:
            self.vertices = load_mesh_vertices('/Volumes/Scratch/Nik_data/MNINonLinear/fsaverage_LR32k/1001_01_MR.L.flat.32k_fs_LR.surf.gii')

        if not self._b:
            self.b = 3 * (self.h[1] - self.h[0])
        
        #make grid
        self.s, self.gridX, self.gridY = grid(self._x, self.vertices, tol=40, spacing=0.4, ch=self.ch)

    def __call__(self, n=1):
        """
        Generate new surrogate map(s).
        
        Parameters
        ----------
        n : int, default 1
            Number of waveshuffled surrogate maps to generate
        
        Returns
        -------
        (n,N) np.ndarray
            Generated map(s) with matched spatial autocorrelation using wavelets
        
        """
        
        rs = np.arange(n)
        surrs = np.row_stack(
            Parallel(self._n_jobs, prefer='threads')(
                delayed(self._call_method)(rs=i) for i in rs
            )
        )
        return np.asarray(surrs.squeeze())

    def _call_method(self, rs=None):
        """
        Subfunction used by .__call__() for parallelization purposes 
        """        
        #generate waveshuffled map
        x_perm = self.waveamp2()
        
        #de-mean
        x_perm = x_perm - np.nanmean(x_perm)
        
        if self._ismasked:
            return np.ma.masked_array(
                data=x_perm, mask=np.isnan(x_perm)).squeeze()
        
        return x_perm.squeeze()

    def compute_variogram(self, x, idx):
        """
        Compute variogram of 'x' using pairs of regions indexed by 'idx'.
        
        Parameters
        ----------
        x : (N,) np.ndarray
            Brain map
        idx : (ns,) np.ndarray[int]
            Indices of randomly sampled brain regions
        
        Returns
        -------
        v : (ns,ns) np.ndarray
            Variogram y-coordinates, i.e. 0.5 * (x_i - x_j) ^ 2, for i,j in idx
        
        """
        diff_ij = x[idx][:, None] - x[self._index[idx, :]]
        return 0.5 * np.square(diff_ij)

    def smooth_variogram(self, u, v, return_h=False):
        """
        Smooth a variogram.
        
        Parameters
        ----------
        u : (N,) np.ndarray
            Pairwise distances, i.e., variogram x-coordinates
        v : (N,) np.ndarray
            Squared differences, i.e., variogram y-coordinates
        return_h : bool, default False
            Return distances at which smooth variogram is computed
        
        Returns
        -------
        (nh,) np.ndarray
            Smoothed variogram samples
        (nh,) np.ndarray
            Distances at which smoothed variogram was computed (returned if
                                                                'return_h' is True)
        
        """
        if len(u) != len(v):
            raise ValueError("u and v must have the same number of elements")
        
        # Subtract each element of h from each pairwise distance `u`.
        # Each row corresponds to a unique h.
        du = np.abs(u - self._h[:, None])
        w = np.exp(-np.square(2.68 * du / self._b) / 2)
        denom = np.nansum(w, axis=1)
        wv = w * v[None, :]
        num = np.nansum(wv, axis=1)
        output = num / denom
        if not return_h:
            return output
        return output, self._h

    def waveamp2(self):
        """
        Returns
        -------
        surr : (N,) np.ndarray
            Surrogate brain map
        
        """
        s = self.s
        gridX = self.gridX
        gridY = self.gridY 
        n = self._scales
        wv = self.wv
        #B = self._B
        #S = self.seed
        
        rr, col = s.shape
        #dim = 1
        N = max(n)+1
        
        if len(s.shape) > 2:
            raise ValueError("Grid must be 2-D (i.e. flatmap)")
        
        #define random shuffling
        #st = np.floor(np.random.choice(1, len(n))*10^6)
        # fnd = np.where(np.ones((rr, col)))
        
        # #find where edges of data are
        # fnd = np.intersect1d(fnd, np.where(s))
        
        #perform wavelet decomposition
        C = pywt.wavedec2(s, wv, mode='zero', level=N)
        CC = [C[0]]
        
        #cnt = 0
        
        for i in np.arange(1, N+1):
            ch, cv, cd = C[i]
            ch_shuff = shuffle_matrix(ch)
            cv_shuff = shuffle_matrix(cv)
            cd_shuff = shuffle_matrix(cd)
        
            cc = (ch_shuff, cv_shuff, cd_shuff)
        
            CC.append(cc)
            
        #recover waveshuffled grid
        ff = pywt.waverec2(CC, wv, mode='zero')
        f = np.zeros((s.shape))
        #f[np.where(s)] = ff[np.where(s)]
            
        if self.ampadj is True:
            #perform amplitude adjustment (force values of ff to equal values of s)
            f_vec = ff.flatten()
            s_vec = s.flatten()
            
            z = np.zeros((len(f_vec), 3))
            s_st = np.sort(s_vec)
            z[:, 0] = f_vec
            z[:, 1] = np.arange(len(f_vec))
            z = z[np.argsort(z[:, 0])]
            z[:, 2] = s_st
            z = z[np.argsort(z[:, 1])]
            f_vec = z[:, 2]
            
            #f[np.where(s)] = f_vec
            
            f = f_vec.reshape(f.shape)
        
        #transform back to mesh
        meshsurr = mesh(f, self.vertices, gridX, gridY)
        
        return meshsurr
    
    def sample(self):
        """
        Randomly sample (without replacement) brain areas for variogram
        computation.

        Returns
        -------
        (self.ns,) np.ndarray
            Indices of randomly sampled areas

        """
        return self._rs.choice(
            a=self._nmap, size=self._ns, replace=False).astype(np.int32)

    @property
    def x(self):
        """ (N,) np.ndarray : brain map scalars """
        if self._ismasked:
            return np.ma.copy(self._x)
        return np.copy(self._x)
    
    @x.setter
    def x(self, x):
        self._ismasked = False
        x_ = dataio(x)
        check_map(x=x_)
        mask = np.isnan(x_)
        if mask.any():
            self._ismasked = True
            brain_map = np.ma.masked_array(data=x_, mask=mask)
        else:
            brain_map = x_
        self._x = brain_map
    
    @property
    def D(self):
        """ (N,N) np.memmap : Pairwise distance matrix """
        return np.copy(self._D)
    
    @D.setter
    def D(self, x):
        x_ = dataio(x)
        n = self._x.size
        if x_.shape[0] != n:
            raise ValueError(
                "D size along axis=0 must equal brain map size")
        self._D = x_[:, 1:self._knn + 1]  # prevent self-coupling
    
    @property
    def index(self):
        """ (N,N) np.memmap : indexes used to sort each row of dist. matrix """
        return np.copy(self._index)
    
    @index.setter
    def index(self, x):
        x_ = dataio(x)
        n = self._x.size
        if x_.shape[0] != n:
            raise ValueError(
                "index size along axis=0 must equal brain map size")
        self._index = x_[:, 1:self._knn+1].astype(np.int32)
    
    @property
    def nmap(self):
        """ int : length of brain map """
        return self._nmap
    
    @nmap.setter
    def nmap(self, x):
        self._nmap = int(x)
    
    @property
    def pv(self):
        """ int : percentile of pairwise distances at which to truncate """
        return self._pv
    
    @pv.setter
    def pv(self, x):
        pv = check_pv(x)
        self._pv = pv
    
    @property
    def deltas(self):
        """ np.ndarray or List[float] : proportions of nearest neighbors """
        return self._deltas
    
    @deltas.setter
    def deltas(self, x):
        check_deltas(deltas=x)
        self._deltas = x
    
    @property
    def nh(self):
        """ int : number of variogram distance intervals """
        return self._nh
    
    @nh.setter
    def nh(self, x):
        self._nh = x
    
    @property
    def knn(self):
        """ int : number of nearest neighbors included in distance matrix """
        return self._knn
    
    @knn.setter
    def knn(self, x):
        if x > self._nmap:
            raise ValueError('knn must be less than len(X)')
        self._knn = int(x)
        
    @property
    def ns(self):
        """ int : number of randomly sampled regions used to construct map """
        return self._ns

    @ns.setter
    def ns(self, x):
        self._ns = int(x)
    
    @property
    def b(self):
        """ numeric : Gaussian kernel bandwidth """
        return self._b
    
    @b.setter
    def b(self, x):
        self._b = x
    
    @property
    def h(self):
        """ np.ndarray : distances at which variogram is evaluated """
        return self._h
    
    @h.setter
    def h(self, x):
        self._h = x
    
    @property
    def scales(self):
        return self._scales
    
    @scales.setter
    def scales(self, x):
        self._scales = x
    
    @property
    def wv(self):
        return self._wv
    
    @wv.setter
    def wv(self, x):
        self._wv = x
    
    @property
    def B(self):
        return self._B
    
    @B.setter
    def B(self, x):
        self._B = x

        