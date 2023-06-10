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

cmap = plt.get_cmap('bone_r')
fontcolor = 'black'

from neuroshape.nulls.eigensphere import eigenmode_resample

def test_resampling(surface, evals, emodes, n=201, decomp_method='matrix'):
    """
    Compute a single eigensphere resampled surrogate at modes = `n`
    """
    # test resampling
    new_vertices = eigenmode_resample(surface, evals, emodes, decomp_method=decomp_method)
    
    #plot_surface(new_surface)

    return new_vertices

def test_surrogates(surface_filename, n=201, num_surrogates=100):
    """
    Compute a number of eigensphere resampled surrogates at modes = `n`
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
    new_surfaces = np.zeros((num_surrogates, coords.shape[0], coords.shape[1]))
    
    for i in range(num_surrogates):
        new_surfaces[i] = test_resampling(surface, evals, emodes, n=200, decomp_method='regression')
    
    return new_surfaces

def test_plot_surfaces(surface, new_surfaces, n=200, data=None, hemi='left', view='lateral', vmin=None, vmax=None, cmap='bone_r', show=True):
    # plot a number of new surfaces and compare to the original
    fig = plt.figure(figsize=(20, 7), constrained_layout=False)
    grid = gridspec.GridSpec(
        1, 4, left=0., right=1., bottom=0., top=1.0,
        height_ratios=[1], width_ratios=[1,1,1,1],
        hspace=0.0, wspace=0.0)
    
    cmap = plt.get_cmap(cmap)
    
    i = 0
    # plot original surface
    orig_vertices = surface.darrays[0].data
    orig_faces = surface.darrays[1].data
    orig_mesh = (orig_vertices, orig_faces)
    
    if data:
        vmin = np.min(data)
        vmax = np.max(data)
    
    ax = fig.add_subplot(grid[i], projection='3d')
    plotting.plot_surf(orig_mesh, data, view=view, vmin=vmin, vmax=vmax,
                       cmap=cmap, avg_method='mean', 
                       axes=ax)
    
    ax = fig.add_subplot(grid[i])
    ax.axis('off')
    ax.text(0.5, 0.1, 'Original surface', ha='center', fontdict={'fontsize':30})
    
    # add title
    label = f'Resampled surfaces at {n} modes'
    ax = fig.add_subplot(grid[i+1])
    ax.axis('off')
    ax.text(1., 0.9, label, ha='center', fontdict={'fontsize':30})
    
    i += 1
    
    surfs = np.random.randint(0, new_surfaces.shape[0], 3)
    
    for surf in surfs:
        # new mesh
        mesh = (new_surfaces[surf], orig_faces)
        
        ax = fig.add_subplot(grid[i], projection='3d')
        plotting.plot_surf(mesh, data, view=view, vmin=vmin,
                           vmax=vmax, cmap=cmap, avg_method='mean',
                           axes=ax)
        
        ax = fig.add_subplot(grid[i])
        ax.axis('off')
        ax.text(0.5, 0.1, f'Resampled surface {surf}', ha='center', fontdict={'fontsize':30})
        
        i += 1
        
    # if data plot colorbar
    if data:
        cax = plt.axes([1.01, 0.3, 0.03, 0.3])
        cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
        cbar.set_ticks([])
        cbar.ax.set_title(f'{vmax:.2f}', fontdict={'fontsize':30, 'color':fontcolor}, pad=20)
        cbar.ax.set_xlabel(f'{vmax:.2f}', fontdict={'fontsize':30, 'color':fontcolor}, labelpad=20)
        
    if show is True:
        plt.show()
        
    return fig, ax
    
    
    
    
    
    
    
    
    
    
    
    
    
    