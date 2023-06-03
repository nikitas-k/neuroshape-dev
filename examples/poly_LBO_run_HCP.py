#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run shapeDNA
"""

from nilearn import plotting, image, masking  
import matplotlib.pyplot as plt
#from nilearn import datasets
from nilearn import surface
from matplotlib import gridspec
import numpy as np
import pyvista as pv
from tqdm import tqdm
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize
import nibabel as nib
import matplotlib.pyplot as plt

from neuroshape.nipype.interfaces.workbench.metric import MetricResample, MetricGradient
import os
from numpy import inf
from neuroshape.poly_eigenmaps import PolyLBO
from neuroshape.permutation import Permutation
from neuroshape.utils.dataio import dataio, save, run, fname_presuffix, split_filename
from pathlib import Path
import os.path as op
from neuromaps.datasets.atlases import fetch_atlas
from neuromaps.stats import permtest_metric
from lapy import TriaMesh
from lapy.ShapeDNA import compute_shapedna 
from neuromaps import transforms
#from neuroshape.nulls.null_parallel import null_parallel
from neuromaps import nulls
from neuroshape.utils import eigen

#get current environment
env = os.environ

cmap = plt.get_cmap('viridis')

# =============================================================================
# #### Path sourcing ####
# =============================================================================

#codeFolder = '/Volumes/Scratch/functional_integration_psychosis/code/subcortex'
#resultFolder = f'{codeFolder}/result/HCP-EP-striatum'
#meshFolder = f'{codeFolder}/meshes'
surfFolder = '/Volumes/Scratch/functional_integration_psychosis/preprocessed/HCP/LBO/surfaces'

baseFolder = "/Volumes/Scratch/functional_integration_psychosis/preprocessed/HCP/LBO"

#fsFolder = "/Volumes/Scratch/functional_integration_psychosis/preprocessing/HCP-EP/FS/fmriresults01"

# =============================================================================
# #### Load and calculate group average LBO/plot ####
# =============================================================================

# fig = plt.figure(figsize=(15,15), constrained_layout=False)
# grid = gridspec.GridSpec(
#     2, 2, left=0., right=1., bottom=0., top=1.,
#     height_ratios=[1.,1.], width_ratios=[1.,1.],
#     hspace=0.0, wspace=0.0)

#hemi = 'L'
common_space = 'fsLR'
common_space_name = '32k_fs_LR'
surf = 'pial'
subject_space = 'native'
type_surface = 'surf'
type_metric = 'func'

# if hemi == 'left':
#     infl_mesh = str(fslr.inflated[0])
# else:
#     infl_mesh = str(fslr.inflated[1])

grp_folder = 'cohorts'
#grps = ['HC','P']
#grps_label = ['Healthy Cohort (HC)', 'Early Psychosis (EP)']

grp = 'HC'

for hemi in ['L','R']:
    #load in files
    text_file = f'{baseFolder}/{grp_folder}/{grp}/subjects.txt'
    subjects = np.loadtxt(text_file, dtype='str').squeeze()
    hemi_files = [f'{surfFolder}/{subject}.{hemi}.{surf}.native.surf.gii' for subject in subjects]
    hemi_data = [nib.load(file).agg_data() for file in hemi_files]
    pth = f'{baseFolder}/{grp_folder}/{grp}/'
    
    #initialize Shape object
    shape = PolyLBO(n_jobs=40, eigs=200)
    
    # #compute eigenmaps
    evs = shape(hemi_data)
    
    resampled_files = []
    
    for ii in range(len(hemi_data)):
        _, fname, ext = split_filename(hemi_files[ii])
        out_file = op.join(pth,
                           'ev_200.' +
                           fname[:-len(type_surface)] +
                           type_metric +
                           ext)
        
        if not Path(out_file).exists():
            #save file
            out = save(evs[ii], out_file)
        
        resampled_file = op.join(out_file[:2+len(hemi)-len(surf)-len(type_metric)-len(ext)] + 
                                 f'.{common_space_name}' +
                                 f'.{type_metric}' +
                                 ext)
        
        if not Path(resampled_file).exists():
            #resample file
            print('Resampling metric file...')
            metrics = MetricResample()
            metrics.inputs.in_file = out_file
            metrics.inputs.method = 'BARYCENTRIC'
            metrics.inputs.current_sphere = f'{surfFolder}/{subjects[ii]}.{hemi}.sphere.native.surf.gii'
            metrics.inputs.new_sphere = f'{surfFolder}/{subjects[ii]}.{hemi}.sphere.{common_space_name}.surf.gii'
            metrics.inputs.largest = True
            metrics.inputs.out_file = resampled_file
            run(metrics.cmdline, env=dict(env))
        
        #keep track of files to reload to calculate average
        resampled_files.append(resampled_file)

    #reload average evs
    evs_all = dataio(resampled_files)
    if hemi == 'L':
        evs_all_lh = evs_all
    else:
        evs_all_rh = evs_all   
        
Psi_lh = evs_all_lh.reshape((282,200,32492))
Psi_rh = evs_all_rh.reshape((282,200,32492))

for sub in range(Psi_lh.shape[0]):
    for j in range(Psi_lh.shape[1]):
        Psi_lh[sub,j] = (Psi_lh[sub,j]-Psi_lh[sub,j].mean())/(Psi_lh[sub,j].std())
        Psi_rh[sub,j] = (Psi_rh[sub,j]-Psi_rh[sub,j].mean())/(Psi_rh[sub,j].std())

np.save('data/HCP_ev_stdized_lh.npy',Psi_lh)
np.save('data/HCP_ev_stdized_rh.npy',Psi_rh)

