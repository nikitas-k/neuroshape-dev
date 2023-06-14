import numpy as np
from joblib import Parallel, delayed
from brainsmash.utils.dataio import dataio
from pathlib import Path
from lapy.ShapeDNA import compute_shapedna
from lapy import TriaMesh
from neuroshape.utils.checks import is_string_like
from neuroshape.utils.eigen import compute_eigenmodes, maximise_recon_metric
from neuromaps.stats import compare_images
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm

__all__ = ['Recon']

recon_types = [
    'up',
    'short',
    'long',    
    ]

cmap = plt.get_cmap('viridis')

class Recon:
    """
    Description
    -----------
    Attempts to reconstruct the given map `y` using LBO eigenmodes `emodes`.
    If `emodes` do not exist, computes them.
    
    Reconstructs the given map `y` by fitting a weighted sum of `emodes` in a
    GLM-like fashion (i.e., `y` = sum( `beta` * `emodes` ) + error ), using either
    LU decomposition to solve the normal equation, or if the solution cannot
    be inverted then uses linear least-squares to calculate the coefficients
    `beta`.
    
    Uses the coefficients `beta` to reconstruct the original data `y` and provides
    an output of reconstructions and reconstruction accuracies.
    
    Parameters
    ----------
        See brainsmash.utils.dataio()
        surface    : list_like, array_like, nib.GiftiImage, string_like or file_like
            
                     if array_like : tuple ((n_vertices,3),(n_faces,3)) of np.ndarrays
                  
                     if nib.GiftiImage : must have darrays of tuple ((n_vertices, 3), n_faces, 3))
            
                     if string_like : string containing path of file to be loaded into
                     memory. Must have .gii extension, other extensions (e.g., .vtk
                     meshes) are future implementations.
            
                     if file-like : string containing paths of files to be loaded into
                     memory. Must have .gii extension, other extensions 
                     (e.g., .vtk meshes) are future implementations.
            
        y          : data map to reconstruct of type np.ndarray, file-like, list-like,
                     or string-like. 
                     Must have the same number of vertices as `surface`
                  
        emodes     : pre-computed eigenmodes, if available, with the same number of
                     vectors as `surface` has vertices, i.e., computed on
                     `surface`. If None, this function computes them based on `surface`
                     using lapy.ShapeDNA
                  
        n_modes    : if `emodes` is None, how many modes to compute using shapeDNA.
                     default is 200.
                  
        nonzero    : bool, flag to compute reconstruction using only the first
                     `n_modes` nonzero modes (ignore the zeroth mode). default True
                  
        n_procs    : number of workers to use in `Parallel` for computation. Default is 1.
        
        metric     : which metric to use for reconstruction accuracy. Currently only 'corr'
                     is implemented.
                  
        type_recon : which reconstruction type to calculate, accepts 'up', 'short',
                     'long'. default 'up'.
                     
                    Description
                    -----------
                    'up'    : compute reconstruction accuracy by starting from
                              zero modes and increasing modes up to `n_modes`
                    
                    'short' : compute reconstruction accuracy by removing short
                              wavelength modes from `n_modes` down to 0
                    
                    'long'  : compute reconstruction accuracy by removing long
                              wavelength modes from `n_modes` down to 0
                     
        maximise   : bool, flag to maximise the reconstruction accuracy by swapping
                     eigenmodes within eigengroups. default True
        
    """
    
    def __init__(self, y, surface=None, emodes=None, n_modes=200, nonzero=True,
                     n_procs=1, metric='corr', type_recon='up', maximise=True):
        
        self._n_jobs = n_procs
        if self._n_jobs > 1:
            self.parallel_flag = True
            
        self._data = y
        
        # try load surface
        if surface:
            self._surface = surface
            self._vertices, self._coords = dataio(self._surface)
        
        # if emodes is given
        if emodes:
            evals, emodes = compute_eigenmodes(surface, num_modes, nonzero)
            
        # copy evals and emodes
        self._emodes = emodes
        self._evals = evals
                
        self._n = self._emodes.shape[1]
        self._metric = metric
        self._type_recon = type_recon
        self._maximise = maximise
        
    def __call__(self, type_recon):
        # check if call is in types
        if type_recon not in recon_types and len(type_recon) > 1:
            raise ValueError("Reconstruction type to compute must be 'up', 'short', or 'long'")            
        
        self._type = type_recon
        
        if self._type == 'up':
            self._function = add_modes
            modes = []
            for x in range(1, self._n+1):
                modes.append(np.arange(x))
            self._modes = modes
            
            if self.parallel_flag is True:
                recons = np.vstack(
                            Parallel(self._n_jobs, prefer='threads')(
                                delayed(self._recon_method)(emodes=x) for x in emodes[self._modes:]))
            
            else:
                recons = np.vstack(_recon_method(emodes=x) for x in emodes[self._modes:])
            
        if self._type == 'short':
            self._function = rem_short_modes
            modes = []
            for x in range(1, self._n+1):
                modes.append(-np.arange(x))
            self._modes = modes
            
            if self.parallel_flag is True:
                recons = np.vstack(
                            Parallel(self._n_jobs, prefer='threads')(
                                delayed(self._recon_method)(emodes=x) for x in emodes[self._modes:]))
            
            else:
                recons = np.vstack(_recon_method(emodes=x) for x in emodes[self._modes:])
            
        if self._type == 'long':
            self._function = rem_long_modes
            modes = []
            for x in range(1, self._n+1):
                modes.append(-np.arange(x))
            self._modes = modes
            
            if self.parallel_flag is True:
                recons = np.vstack(
                            Parallel(self._n_jobs, prefer='threads')(
                                delayed(self._recon_method)(emodes=x) for x in emodes[:self._modes]))
            
            else:
                recons = np.vstack(_recon_method(emodes=x) for x in emodes[:self._modes])
                    
        self._recons = recons
            
        
    def _recon_method(self, emodes):
        """
        Subfunction used by .__call__() for parallelization purposes
        """
        y = self._data
        metric = self._metric
        
        if self._maximise is True:
            recon = maximise_recon_metric(emodes, y, metric)
            
        else:
            betas = np.matmul(emodes.T, y)
            recon = np.matmul(betas.T, emodes).reshape(-1,)
        
        return recon
    
    @property
    def surface(self):
        return self._surface
    
    @surface.setter
    def surface(self, x):
        try:
            x_ = dataio(x)
        except:
            raise ValueError("Check surface")
        
        self._surface = x_
        self._vertices, self._faces = x_
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, x):
        try:
            x_ = dataio(x)
        except:
            raise ValueError("Could not load data, check input")
            
        if x_.shape[0] != self._vertices.shape[0]:
            raise ValueError("New data must have the same number of vertices as the currently loaded surface. Set the new surface first before setting the new data")
        self._data = x_
    
    @property
    def plot_recon(self):
        """plot reconstruction figure and axes handles"""
        return self.plot_recon
    
    @plot.setter
    def plot_recon(self, x, n, hemi='left', view='lateral'):
        if x is not None and x != self._type:
            try:
                print("New recon type, computing reconstructions for type {}".format(str(x)))
                recons = self.__call__(x)
            except (ValueError):
                raise ValueError("Reconstruction type to plot must be 'up', 'short', or 'long'")
        else:
            recons = self._recons
        
        if self._recon_corr is None:
            print("Reconstruction accuracy has not been calculated, computing for data given when initialized")
            corr = self.recon_accuracy()
        
        if n not in range(recons.shape[0]):
            raise ValueError("Cannot plot reconstruction")
            
        # now plot data
        fig = plt.figure(figsize=(15,9), constrained_layout=False)
        grid = gridspec.GridSpec(
            1, 2, left=0., right=1., bottom=0., top=1.,
            height_ratios=1., width_ratios=[1.,1.],
            hspace=0.0, wspace=0.0)
            
        mesh = (self._vertices, self._faces)
        data = self._data
        
        cmap = self._cmap
        vmin = np.min(data)
        vmax = np.max(data)
        
        colorbar = False
        
        ax = fig.add_subplot(grid[0], projection='3d')
        plotting.plot_surf(mesh, surf_map=data, hemi=hemi,
                           view=view, vmin=vmin, vmax=vmax,
                           colorbar=colorbar, cmap=cmap,
                           axes=ax)
        ax.dist = 7
        # label
        ax = fig.add_subplot(grid[0])
        ax.axis('off')
        ax.text(0.5, 0.5, "Original map", ha="center", fontdict={'fontsize':30})
        
        # now plot recon
        ax = fig.add_subplot(grid[1], projection='3d')
        plotting.plot_surf(mesh, surf_map=recons[n], hemi=hemi,
                           view=view, vmin=vmin, vmax=vmax,
                           colorbar=colorbar, cmap=cmap,
                           axes=ax)
        ax.dist = 7
        #label
        ax = fig.add_subplot(grid[1])
        ax.axis('off')
        ax.text(0.5, 0.5, "Reconstruction", ha="center", fontdict={'fontsize':30})
        
        # now colorbar
        cax = plt.axes([0.8, 0.32, 0.03, 0.4])
        cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
        cbar.set_ticks([])
        cbar.ax.set_title('max', fontdict={'fontsize':30}, pad=20)
        cbar.ax.set_xlabel('min', fontdict={'fontsize':30}, labelpad=20)
        
        plt.show()
        
        
    @property
    def plot_accuracy(self):
        return self._plot_acc
    
    @plot_accuracy.setter
    def plot_accuracy(self, x):
        if x is not None and x != self._type:
            try:
                print("New recon type, computing reconstructions for type {}".format(str(x)))
                recons = self.__call__(x)
            except (ValueError):
                raise ValueError("Reconstruction type to plot must be 'up', 'short', or 'long'")
    
        else:
            recons = self._recons
        
        if self._recon_corr is None:
            print("Reconstruction accuracy has not been calculated, computing for data given when initialized")
            corr = self.recon_accuracy()
            
        # now plot recon accuracy
        fig = plt.figure(figsize=(15, 9), constrained_layout=False)
        ax = fig.add_subplot()
        
        n = self._n
        x = np.arange(1, n+1)
        per_x = 1/x * 100
        if self._type == 'up':
            title = "Reconstruction accuracy adding modes from zero to {} modes".format(str(n))
        if self._type == 'short':
            title = "Reconstruction accuracy removing short-wavelength modes from {} to zero modes".format(str(n))
        if self._type == 'long':
            title = "Reconstruction accuracy removing long-wavelength modes from {} to zero modes".format(str(n))
        
        ax.plot(per_x, corr, 'b')
        ax.ylabel("Reconstruction accuracy")
        ax.xlabel("Proportion of modes")
        ax.xticklabels(np.arange(0, 100, 5))
        
        self._plot_acc = fig, ax
        
        plt.show()
            
        
    @property
    def recon_accuracy(self):
        """Reconstruction accuracy from a set of reconstructions"""
        return self._recon_corr
    
    @recon_accuracy.setter
    def recon_accuracy(self, x):
        corr = np.zeros(self._recons.shape[0])
        if x is not None:
            try:
                corr[i] = np.hstack(compare_images(self._recons[i], x))
                self._recon_corr = corr
            except:
                e = "Could not compute reconstruction accuracy, check input data (check if number of vertices are the same)"
                raise ValueError(e)
                
        else:
            corr[i] = np.hstack(compare_images(self._recons[i], self._data))
            
        self._recon_corr = corr
        
    @property
    def cmap(self):
        """colormap for plotting"""
        return self._cmap
    
    @cmap.setter
    def cmap(self, x):
        if not is_string_like(x):
            raise ValueError("Colormap must be string")
        try:
            cmap = plt.get_cmap(x)
        except:
            raise ValueError("Colormap must be in matplotlib recognized list")
        
        self._cmap = cmap

    