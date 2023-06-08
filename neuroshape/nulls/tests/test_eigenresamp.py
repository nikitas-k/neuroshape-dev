# -*- coding: utf-8 -*-
"""
For testing neuroshape.nulls.eigensphere functionality
"""

import numpy as np
import nibabel as nib
from lapy import TriaMesh
from lapy.ShapeDNA import compute_shapedna

from neuroshape.nulls.eigensphere import eigenmode_resample

def test_resampling(surface_filename, n=200):
    
    surface = nib.load(surface_filename)
    
    coords, faces = surface.darrays
    
    coords = coords.data
    faces = faces.data
    
    # compute LBO
    tria = TriaMesh(coords, faces)
    ev = compute_shapedna(tria, k=n)
    
    # exclude the zero mode
    emodes = ev['Eigenvectors'][:,1:]
    emodes = emodes/np.linalg.norm(emodes, axis=0)
    evals = ev['Eigenvalues']
    
    # test resampling
    new_surface = eigenmode_resample(surface, evals, emodes)

    return new_surface