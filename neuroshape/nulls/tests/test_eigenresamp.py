# -*- coding: utf-8 -*-
"""
For testing neuroshape.nulls.eigensphere functionality
"""

import numpy as np
import nibabel as nib
from lapy import TriaMesh
from lapy.ShapeDNA import compute_shapedna
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm
from nilearn import plotting
from neuroshape.utils.normalize_data import normalize_data

cmap = plt.get_cmap('bone_r')
fontcolor = 'black'

from neuroshape.nulls.eigensphere import eigenmode_resample

def test_resampling(surface, data, evals, emodes, n=201, decomp_method='matrix'):
    """
    Compute a single eigensphere resampled surrogate at modes = `n`
    """
    # test resampling
    surrogate = eigenmode_resample(surface, data, evals, emodes, decomp_method=decomp_method)
    
    #plot_surface(new_surface)

    return surrogate


def test_surrogates(surface_filename, data, n=201, num_surrogates=100, decomp_method='matrix'):
    """
    Compute a number of eigensphere resampled surrogates at modes = `n`
    and create surrogate data by reconstruction
    """
    # load surface
    surface = nib.load(surface_filename)
    
    # make TriaMesh
    coords, faces = surface.darrays
    
    coords = coords.data
    faces = faces.data
    
    # compute LBO
    tria = TriaMesh(coords, faces)
    ev = compute_shapedna(tria, k=n)
    
    # exclude the zeroth mode
    emodes = ev['Eigenvectors'][:,1:]
    emodes = emodes/np.linalg.norm(emodes, axis=0)
    evals = ev['Eigenvalues'][1:]
    
    # initialize the surrogate array - compute only the vertices, the faces are the same
    surrogates = np.zeros((num_surrogates, data.shape[0]))
    
    for i in range(num_surrogates):
        surrogates[i] = test_resampling(surface, data, evals, emodes, n=n, decomp_method=decomp_method)
    
    return surrogates

def test_plot_surrogates(surface, data, surrogates, n=201, hemi='left', view='lateral', cmap='viridis', show=True):
    # plot a number of new surfaces and compare to the original
    fig = plt.figure(figsize=(23, 10), constrained_layout=False)
    grid = gridspec.GridSpec(
        2, 5, left=0., right=1., bottom=0., top=1.0,
        height_ratios=[0.6,.5], width_ratios=[1.,1.,1.,1.,1.],
        hspace=0.0, wspace=0.0)
    
    cmap = plt.get_cmap(cmap)
    
    i = 0
    # plot original surface
    vertices = surface.darrays[0].data
    faces = surface.darrays[1].data
    mesh = (vertices, faces)
    
    data_norm = normalize_data(data)
    
    vmin = np.min(data_norm)
    vmax = np.max(data_norm)
    
    ax = fig.add_subplot(grid[i], projection='3d')
    plotting.plot_surf(mesh, data_norm, view=view, vmin=vmin, vmax=vmax,
                       cmap=cmap, avg_method='mean', 
                       axes=ax)
    
    ax = fig.add_subplot(grid[i])
    ax.axis('off')
    ax.text(0.5, 0.1, 'Original map', ha='center', fontdict={'fontsize':30})
    
    # add title
    label = f'Resampled maps at {n-1} modes'
    ax = fig.add_subplot(grid[i+1])
    ax.axis('off')
    ax.text(1.5, 1., label, ha='center', fontdict={'fontsize':30})
    
    # normalize surrogates for plotting
    surrogates_norm = normalize_data(surrogates)
    
    i += 1
    
    for surr in range(surrogates_norm.shape[0]-1):
        ax = fig.add_subplot(grid[i], projection='3d')
        plotting.plot_surf(mesh, surrogates_norm[surr], view=view, vmin=vmin,
                           vmax=vmax, cmap=cmap, avg_method='mean',
                           axes=ax)
        
        ax = fig.add_subplot(grid[i])
        ax.axis('off')
        ax.text(0.5, 0.1, f'Resampled map {surr+1}', ha='center', fontdict={'fontsize':30})
        
        i += 1

    # colorbar
    cax = plt.axes([1.04, 0.3, 0.03, 0.6])
    cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
    cbar.set_ticks([])
    cbar.ax.set_title(f'{vmax:.2f}', fontdict={'fontsize':30, 'color':fontcolor}, pad=20)
    cbar.ax.set_xlabel(f'{vmin:.2f}', fontdict={'fontsize':30, 'color':fontcolor}, labelpad=20)
        
    if show is True:
        plt.show()
        
    return fig, ax
    
    
    
    
    
    
    
    
    
    
    
    
    
    