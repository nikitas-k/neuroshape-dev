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
from scipy.interpolate import interp1d
from neuromaps import transforms, datasets, images, nulls, resampling
from neuromaps.stats import compare_images
from neuromaps.datasets import fetch_annotation, fetch_atlas
from lapy import ShapeDNA
from lapy import TriaMesh
from scipy.sparse.linalg import eigsh, splu
from nibabel.freesurfer.io import write_morph_data, read_label
from subcortex.functions.write_label import write_label
import subprocess
from nipype.interfaces.workbench import MetricResample
import os
from numpy import inf

cmap = plt.get_cmap('viridis')

# =============================================================================
# #### Path sourcing ####
# =============================================================================

codeFolder = '/Volumes/Scratch/functional_integration_psychosis/code/subcortex'
resultFolder = f'{codeFolder}/result/HCP-EP-striatum'
meshFolder = f'{codeFolder}/meshes'

baseFolder = "/Volumes/Scratch/functional_integration_psychosis/preprocessed/HCP-EP/LBO"

#fsFolder = "/Volumes/Scratch/functional_integration_psychosis/preprocessing/HCP-EP/FS/fmriresults01"

# =============================================================================
# #### Load and calculate group average LBO/plot ####
# =============================================================================

# fig = plt.figure(figsize=(15,15), constrained_layout=False)
# grid = gridspec.GridSpec(
#     2, 2, left=0., right=1., bottom=0., top=1.,
#     height_ratios=[1.,1.], width_ratios=[1.,1.],
#     hspace=0.0, wspace=0.0)

interpolation = 'linear'
kind = 'ball'
radius = 3
n_samples = None
row = 0
mask_img = 'masks/cortex.nii'
cmap = cmap

hemi = 'left'

# if hemi == 'left':
#     infl_mesh = str(fslr.inflated[0])
# else:
#     infl_mesh = str(fslr.inflated[1])

grp_folder = 'cohorts'
grps = ['HC','P']
grps_label = ['Healthy Cohort (HC)', 'Early Psychosis (EP)']

i = 0
Vn = 2

for grp in grps:
    print(f'Compute {grp}-average LBO eigenmaps and magnitudes\n')
    #Initialize arrays
    ev_lh_fsLR_all = np.zeros(32492,)
    ev_rh_fsLR_all = np.zeros(32492,)
    
    #Load surface data
    with open(f'{baseFolder}/{grp_folder}/{grp}/subjects.txt', 'r') as f:
        for line in f:
            subject = line.strip()
            
            if os.path.isfile(f'{baseFolder}/{grp_folder}/HC.R.Vn{Vn}.mag.32k_fs_LR.func.gii') is False:
            
                #left
                surf_lh = nib.load(f'{baseFolder}/{grp_folder}/{grp}/{subject}.L.pial.native.surf.gii').agg_data()
                coords, faces = surf_lh
                tria_lh = TriaMesh(coords, faces)
                ev_lh = ShapeDNA.compute_shapedna(tria_lh, k=3)
                ShapeDNA.normalize_ev(tria_lh, ev_lh['Eigenvectors'], method='geometry')
                ShapeDNA.reweight_ev(ev_lh['Eigenvalues'])
                
                #eigenmaps
                eigvec_lh = ev_lh['Eigenvectors'][:,Vn-1]
                #eigval_lh = ev_lh['Eigenvalues']
                
                eig_lh = nib.GiftiImage()
                eig_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(eigvec_lh))
                
                nib.save(eig_lh, f'{baseFolder}/{grp_folder}/{grp}/{subject}.L.Vn{Vn}.ev.native.func.gii')
                
                
                #right
                surf_rh = nib.load(f'{baseFolder}/{grp_folder}/{grp}/{subject}.R.pial.native.surf.gii').agg_data()
                coords, faces = surf_rh
                tria_rh = TriaMesh(coords, faces)
                ev_rh = ShapeDNA.compute_shapedna(tria_rh, k=3)
                ShapeDNA.normalize_ev(tria_rh, ev_rh['Eigenvectors'], method='geometry')
                ShapeDNA.reweight_ev(ev_rh['Eigenvalues'])
                
                #eigenmaps
                eigvec_rh = ev_rh['Eigenvectors'][:,Vn-1]
                #eigval_lh = ev_lh['Eigenvalues']
                
                eig_rh = nib.GiftiImage()
                eig_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(eigvec_rh))
                
                nib.save(eig_rh, f'{baseFolder}/{grp_folder}/{grp}/{subject}.R.Vn{Vn}.ev.native.func.gii')
                
                #resample metric files to fsLR space
                #left eig
                
                metric_lh = MetricResample()
                metric_lh.inputs.in_file = f'{baseFolder}/{grp_folder}/{grp}/{subject}.L.Vn{Vn}.ev.native.func.gii'
                metric_lh.inputs.method = 'BARYCENTRIC'
                metric_lh.inputs.current_sphere = f'{baseFolder}/{grp_folder}/{grp}/{subject}.L.sphere.native.surf.gii'
                metric_lh.inputs.new_sphere = f'{baseFolder}/{grp_folder}/{grp}/{subject}.L.sphere.32k_fs_LR.surf.gii'
                metric_lh.inputs.largest = True
                metric_lh.inputs.out_file = f'{baseFolder}/{grp_folder}/{grp}/{subject}.L.Vn{Vn}.ev.32k_fs_LR.func.gii'
                os.system(metric_lh.cmdline)
                
                #left mag
                
                # metric_lh = MetricResample()
                # metric_lh.inputs.in_file = f'{baseFolder}/{grp_folder}/{grp}/{subject}.L.Vn{Vn}.mag.native.func.gii'
                # metric_lh.inputs.method = 'BARYCENTRIC'
                # metric_lh.inputs.current_sphere = f'{baseFolder}/{grp_folder}/{grp}/{subject}.L.sphere.native.surf.gii'
                # metric_lh.inputs.new_sphere = f'{baseFolder}/{grp_folder}/{grp}/{subject}.L.sphere.32k_fs_LR.surf.gii'
                # metric_lh.inputs.largest = True
                # metric_lh.inputs.out_file = f'{baseFolder}/{grp_folder}/{grp}/{subject}.L.Vn{Vn}.mag.32k_fs_LR.func.gii'
                # os.system(metric_lh.cmdline)
                
                #right eig
                
                metric_rh = MetricResample()
                metric_rh.inputs.in_file = f'{baseFolder}/{grp_folder}/{grp}/{subject}.R.Vn{Vn}.ev.native.func.gii'
                metric_rh.inputs.method = 'BARYCENTRIC'
                metric_rh.inputs.current_sphere = f'{baseFolder}/{grp_folder}/{grp}/{subject}.R.sphere.native.surf.gii'
                metric_rh.inputs.new_sphere = f'{baseFolder}/{grp_folder}/{grp}/{subject}.R.sphere.32k_fs_LR.surf.gii'
                metric_rh.inputs.largest = True
                metric_rh.inputs.out_file = f'{baseFolder}/{grp_folder}/{grp}/{subject}.R.Vn{Vn}.ev.32k_fs_LR.func.gii'
                os.system(metric_rh.cmdline)
                
                #right mag
                
                # metric_rh = MetricResample()
                # metric_rh.inputs.in_file = f'{baseFolder}/{grp_folder}/{grp}/{subject}.R.Vn{Vn}.mag.native.func.gii'
                # metric_rh.inputs.method = 'BARYCENTRIC'
                # metric_rh.inputs.current_sphere = f'{baseFolder}/{grp_folder}/{grp}/{subject}.R.sphere.native.surf.gii'
                # metric_rh.inputs.new_sphere = f'{baseFolder}/{grp_folder}/{grp}/{subject}.R.sphere.32k_fs_LR.surf.gii'
                # metric_rh.inputs.largest = True
                # metric_rh.inputs.out_file = f'{baseFolder}/{grp_folder}/{grp}/{subject}.R.Vn{Vn}.mag.32k_fs_LR.func.gii'
                # os.system(metric_rh.cmdline)
            
            #load resampled metric files to calculate group average eigenvector and magnitude maps
            ev_lh_fsLR = nib.load(f'{baseFolder}/{grp_folder}/{grp}/{subject}.L.Vn{Vn}.ev.32k_fs_LR.func.gii').agg_data()
            ev_lh_fsLR_all = np.vstack((ev_lh_fsLR_all, ev_lh_fsLR))
            
            # mag_lh_fsLR = nib.load(f'{baseFolder}/{grp_folder}/{grp}/{subject}.L.Vn{Vn}.mag.32k_fs_LR.func.gii').agg_data()
            # mag_lh_fsLR_all = np.vstack((mag_lh_fsLR_all, mag_lh_fsLR))
            
            ev_rh_fsLR = nib.load(f'{baseFolder}/{grp_folder}/{grp}/{subject}.R.Vn{Vn}.ev.32k_fs_LR.func.gii').agg_data()
            ev_rh_fsLR_all = np.vstack((ev_rh_fsLR_all, ev_rh_fsLR))
            
            # mag_rh_fsLR = nib.load(f'{baseFolder}/{grp_folder}/{grp}/{subject}.R.Vn{Vn}.mag.32k_fs_LR.func.gii').agg_data()
            # mag_rh_fsLR_all = np.vstack((mag_rh_fsLR_all, mag_rh_fsLR))
            
    
    
    #derive group average eigenvector magnitude maps and standard deviation for z-score
    if grp == 'HC':
        HC_ev_avg_lh_fsLR = np.nanmean(ev_lh_fsLR_all[1:].T, axis=1).astype('float32')
        HC_ev_avg_rh_fsLR = np.nanmean(ev_rh_fsLR_all[1:].T, axis=1).astype('float32')
        HC_ev_avg_lh_fsLR[HC_ev_avg_lh_fsLR == inf] = 0
        HC_ev_avg_rh_fsLR[HC_ev_avg_rh_fsLR == inf] = 0
        
        
        # HC_mag_avg_lh_fsLR = np.nanmean(mag_lh_fsLR_all.T, axis=1).astype('float32')
        # HC_mag_avg_rh_fsLR = np.nanmean(mag_rh_fsLR_all.T, axis=1).astype('float32')
        
    else:
        P_ev_avg_lh_fsLR = np.nanmean(ev_lh_fsLR_all[1:].T, axis=1).astype('float32')
        P_ev_avg_rh_fsLR = np.nanmean(ev_rh_fsLR_all[1:].T, axis=1).astype('float32')
        P_ev_avg_lh_fsLR[P_ev_avg_lh_fsLR == inf] = 0
        P_ev_avg_rh_fsLR[P_ev_avg_rh_fsLR == inf] = 0
        
    
#save average surface metrics HC

HC_ev_avg_lh_fsLR_img = nib.GiftiImage()
HC_ev_avg_lh_fsLR_img.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(HC_ev_avg_lh_fsLR))
nib.save(HC_ev_avg_lh_fsLR_img, f'{baseFolder}/{grp_folder}/HC.L.Vn{Vn}.ev.32k_fs_LR.func.gii')

HC_ev_avg_rh_fsLR_img = nib.GiftiImage()
HC_ev_avg_rh_fsLR_img.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(HC_ev_avg_rh_fsLR))
nib.save(HC_ev_avg_rh_fsLR_img, f'{baseFolder}/{grp_folder}/HC.R.Vn{Vn}.ev.32k_fs_LR.func.gii')

#magnitudes and vectors

subprocess.run(['wb_command', '-metric-gradient',
                f'{meshFolder}/L.midthickness.32k_fs_LR.surf.gii',
                f'{baseFolder}/{grp_folder}/HC.L.Vn{Vn}.ev.32k_fs_LR.func.gii',
                f'{baseFolder}/{grp_folder}/HC.L.Vn{Vn}.mag.32k_fs_LR.func.gii',
                '-vectors', f'{baseFolder}/{grp_folder}/HC.L.Vn{Vn}.vectors.32k_fs_LR.func.gii'
                ])

subprocess.run(['wb_command', '-metric-gradient',
                f'{meshFolder}/R.midthickness.32k_fs_LR.surf.gii',
                f'{baseFolder}/{grp_folder}/HC.R.Vn{Vn}.ev.32k_fs_LR.func.gii',
                f'{baseFolder}/{grp_folder}/HC.R.Vn{Vn}.mag.32k_fs_LR.func.gii',
                '-vectors', f'{baseFolder}/{grp_folder}/HC.R.Vn{Vn}.vectors.32k_fs_LR.func.gii'
                ])

#save average surface metrics P

P_ev_avg_lh_fsLR_img = nib.GiftiImage()
P_ev_avg_lh_fsLR_img.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(P_ev_avg_lh_fsLR))
nib.save(P_ev_avg_lh_fsLR_img, f'{baseFolder}/{grp_folder}/P.L.Vn{Vn}.ev.32k_fs_LR.func.gii')

P_ev_avg_rh_fsLR_img = nib.GiftiImage()
P_ev_avg_rh_fsLR_img.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(P_ev_avg_rh_fsLR))
nib.save(P_ev_avg_rh_fsLR_img, f'{baseFolder}/{grp_folder}/P.R.Vn{Vn}.ev.32k_fs_LR.func.gii')

#magnitudes and vectors

subprocess.run(['wb_command', '-metric-gradient',
                f'{meshFolder}/L.midthickness.32k_fs_LR.surf.gii',
                f'{baseFolder}/{grp_folder}/P.L.Vn{Vn}.ev.32k_fs_LR.func.gii',
                f'{baseFolder}/{grp_folder}/P.L.Vn{Vn}.mag.32k_fs_LR.func.gii',
                '-vectors', f'{baseFolder}/{grp_folder}/P.L.Vn{Vn}.vectors.32k_fs_LR.func.gii'
                ])

subprocess.run(['wb_command', '-metric-gradient',
                f'{meshFolder}/R.midthickness.32k_fs_LR.surf.gii',
                f'{baseFolder}/{grp_folder}/P.R.Vn{Vn}.ev.32k_fs_LR.func.gii',
                f'{baseFolder}/{grp_folder}/P.R.Vn{Vn}.mag.32k_fs_LR.func.gii',
                '-vectors', f'{baseFolder}/{grp_folder}/P.R.Vn{Vn}.vectors.32k_fs_LR.func.gii'
                ])

#load group average magnitudes
HC_mag_avg_lh_fsLR = nib.load(f'{baseFolder}/{grp_folder}/HC.L.Vn{Vn}.mag.32k_fs_LR.func.gii')
HC_mag_avg_rh_fsLR = nib.load(f'{baseFolder}/{grp_folder}/HC.R.Vn{Vn}.mag.32k_fs_LR.func.gii')

P_mag_avg_lh_fsLR = nib.load(f'{baseFolder}/{grp_folder}/P.L.Vn{Vn}.mag.32k_fs_LR.func.gii')
P_mag_avg_rh_fsLR = nib.load(f'{baseFolder}/{grp_folder}/P.R.Vn{Vn}.mag.32k_fs_LR.func.gii')
    
#create nulls - Spin Test, use your own method as desired

HC_mag_avg_fsLR = (HC_mag_avg_lh_fsLR, HC_mag_avg_rh_fsLR)
P_mag_avg_fsLR = (P_mag_avg_lh_fsLR, P_mag_avg_rh_fsLR)

#combine
rotated_mag_avg_fsLR = nulls.alexander_bloch(HC_mag_avg_fsLR,
                                                atlas='fslr',
                                                density='32k',
                                                n_perm=1000,
                                                seed=1234)

rotated_mag_avg_fsLR_inv = nulls.alexander_bloch(P_mag_avg_fsLR,
                                                    atlas='fslr',
                                                    density='32k',
                                                    n_perm=1000,
                                                    seed=1234)

#calculate z-scores
HC_mag_avg_fsLR_data = np.hstack((HC_mag_avg_fsLR[0].agg_data(), HC_mag_avg_fsLR[1].agg_data()))
P_mag_avg_fsLR_data = np.hstack((P_mag_avg_fsLR[0].agg_data(), P_mag_avg_fsLR[1].agg_data()))

data_ref = HC_mag_avg_fsLR_data - P_mag_avg_fsLR_data
data_ref_inv = P_mag_avg_fsLR_data - HC_mag_avg_fsLR_data

data_all = rotated_mag_avg_fsLR - rotated_mag_avg_fsLR_inv
data_all_inv = rotated_mag_avg_fsLR_inv - rotated_mag_avg_fsLR

mean_data = np.mean(data_all, axis=1)
mean_data_inv = np.mean(data_all_inv, axis=1)

std_all = np.std(data_all, axis=1)
std_all_inv = np.std(data_all_inv, axis=1)

zscore = np.divide((data_ref - mean_data), std_all)
zscore_inv = np.divide((data_ref_inv - mean_data_inv), std_all_inv)

#save images

data_zscore_lh = nib.GiftiImage()
data_zscore_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore[:len(zscore)//2]))
data_zscore_rh = nib.GiftiImage()
data_zscore_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore[len(zscore)//2:]))

nib.save(data_zscore_lh, f'{baseFolder}/{grp_folder}/L.Vn{Vn}.HC-P.zscore.32k_fs_LR.func.gii')
nib.save(data_zscore_rh, f'{baseFolder}/{grp_folder}/R.Vn{Vn}.HC-P.zscore.32k_fs_LR.func.gii')

#inverse
data_zscore_lh_inv = nib.GiftiImage()
data_zscore_lh_inv.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore_inv[:len(zscore_inv)//2]))
data_zscore_rh_inv = nib.GiftiImage()
data_zscore_rh_inv.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore_inv[len(zscore_inv)//2:]))

nib.save(data_zscore_lh_inv, f'{baseFolder}/{grp_folder}/L.Vn{Vn}.P-HC.zscore.32k_fs_LR.func.gii')
nib.save(data_zscore_rh_inv, f'{baseFolder}/{grp_folder}/R.Vn{Vn}.P-HC.zscore.32k_fs_LR.func.gii')

#calculate p-values
pval_sup = np.sum([(i >= data_ref) for i in data_all.T], axis=0)/1000

data_pval_sup_lh = nib.GiftiImage()
data_pval_sup_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pval_sup[:len(zscore)//2].astype('float32')))
data_pval_sup_rh = nib.GiftiImage()
data_pval_sup_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pval_sup[len(zscore)//2:].astype('float32')))

nib.save(data_pval_sup_lh, f'{baseFolder}/{grp_folder}/L.Vn{Vn}.HC-P.pval.32k_fs_LR.func.gii')
nib.save(data_pval_sup_rh, f'{baseFolder}/{grp_folder}/R.Vn{Vn}.HC-P.pval.32k_fs_LR.func.gii')

#inverse
pval_inf = np.sum([(i <= data_ref_inv) for i in data_all_inv.T], axis=0)/1000

data_pval_inf_lh = nib.GiftiImage()
data_pval_inf_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pval_sup[:len(zscore)//2].astype('float32')))
data_pval_inf_rh = nib.GiftiImage()
data_pval_inf_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pval_sup[len(zscore)//2:].astype('float32')))

nib.save(data_pval_inf_lh, f'{baseFolder}/{grp_folder}/L.Vn{Vn}.P-HC.pval.32k_fs_LR.func.gii')
nib.save(data_pval_inf_rh, f'{baseFolder}/{grp_folder}/R.Vn{Vn}.P-HC.pval.32k_fs_LR.func.gii')

#threshold z-scores by pvals

zscore_thr = np.where(pval_inf < 0.05, zscore, 0.) + np.where(pval_sup > 0.95, zscore, 0.)

data_zscore_thr_lh = nib.GiftiImage()
data_zscore_thr_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore_thr[:len(zscore_thr)//2].astype('float32')))
data_zscore_thr_rh = nib.GiftiImage()
data_zscore_thr_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore_thr[len(zscore_thr)//2:].astype('float32')))

nib.save(data_zscore_thr_lh, f'{baseFolder}/{grp_folder}/L.Vn{Vn}.HC-P.zscore_thr.32k_fs_LR.func.gii')
nib.save(data_zscore_thr_rh, f'{baseFolder}/{grp_folder}/R.Vn{Vn}.HC-P.zscore_thr.32k_fs_LR.func.gii')

#left
# data_ref_lh = HC_mag_avg_lh_fsLR - P_mag_avg_lh_fsLR
# data_ref_lh_inv = P_mag_avg_lh_fsLR - HC_mag_avg_lh_fsLR

# data_all_lh = np.subtract((rotated_mag_avg_lh_fsLR, rotated_mag_avg_lh_fsLR_inv), axis=0)
# data_all_lh_inv = np.subtract((rotated_mag_avg_lh_fsLR_inv, rotated_mag_avg_lh_fsLR), axis=0)

# mean_data_lh = np.mean(data_all_lh, axis=1)
# mean_data_lh_inv = np.mean(data_all_lh_inv, axis=1)

# std_all_lh = np.std(data_all_lh, axis=1)
# std_all_lh_inv = np.std(data_all_lh_inv, axis=1)

# zscore_lh = np.divide((np.subtract((data_ref_lh, mean_data_lh)), std_all_lh))
# zscore_lh_inv = np.divide((np.subtract((data_ref_lh_inv, mean_data_lh_inv)), std_all_lh_inv))

# #save images

# nib.save(zscore_lh, f'{baseFolder}/{grp_folder}/L.Vn{Vn}.HC-P.zscore.32k_fs_LR.func.gii')
# nib.save(zscore_lh_inv, f'{baseFolder}/{grp_folder}/L.Vn{Vn}.P-HC.zscore.32k_fs_LR.func.gii')

# #right
# data_ref_rh = HC_mag_avg_rh_fsLR - P_mag_avg_rh_fsLR
# data_ref_rh_inv = P_mag_avg_rh_fsLR - HC_mag_avg_rh_fsLR

# data_all_rh = np.subtract((rotated_mag_avg_rh_fsLR, rotated_mag_avg_rh_fsLR_inv), axis=0)
# data_all_rh_inv = np.subtract((rotated_mag_avg_rh_fsLR_inv, rotated_mag_avg_rh_fsLR), axis=0)

# mean_data_rh = np.mean(data_all_rh, axis=1)
# mean_data_rh_inv = np.mean(data_all_rh_inv, axis=1)

# std_all_rh = np.std(data_all_rh, axis=1)
# std_all_rh_inv = np.std(data_all_rh_inv, axis=1)

# zscore_rh = np.divide((np.subtract((data_ref_rh, mean_data_rh)), std_all_rh))
# zscore_rh_inv = np.divide((np.subtract((data_ref_rh_inv, mean_data_rh_inv)), std_all_rh_inv))

# #save images

# nib.save(zscore_rh, f'{baseFolder}/{grp_folder}/R.Vn{Vn}.HC-P.zscore.32k_fs_LR.func.gii')
# nib.save(zscore_rh_inv, f'{baseFolder}/{grp_folder}/R.Vn{Vn}.P-HC.zscore.32k_fs_LR.func.gii')
    

#calculate z-score
    
    
    #save images
                        
            
#     lgi_avg_lh_texture = np.divide(lgi_avg_lh_texture, ndata)        
#     lgi_avg_rh_texture = np.divide(lgi_avg_rh_texture, ndata)
    
#     # calculate z-score
    
#     avg_lh = nib.GiftiImage()
#     avg_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(np.float32(lgi_avg_lh_texture)))
    
#     avg_rh = nib.GiftiImage()
#     avg_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(np.float32(lgi_avg_rh_texture)))
    
#     #save group average lgi as gifti
#     nib.save(avg_lh, f'{baseFolder}/{grp_folder}/{grp}.lh.pial_lgi.shape.gii')
#     nib.save(avg_rh, f'{baseFolder}/{grp_folder}/{grp}.rh.pial_lgi.shape.gii')
    
#     #lgi = (lgi_lh, lgi_rh)
    
#     for view in ['lateral','medial']:
#         if hemi == 'left':
#             ax = fig.add_subplot(grid[i], projection='3d')
#             vmin = min(lgi_avg_lh_texture)
#             vmax = max(lgi_avg_lh_texture)
#             plotting.plot_surf(
#                 infl_mesh, lgi_avg_lh_texture, hemi=hemi, view=view,
#                 colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
#                 axes=ax)
#             ax.dist = 7
            
#         else:
#             ax = fig.add_subplot(grid[i], projection='3d')
#             vmin = min(lgi_avg_rh_texture)
#             vmax = max(lgi_avg_rh_texture)
#             plotting.plot_surf(
#                 infl_mesh, lgi_avg_rh_texture, hemi=hemi, view=view,
#                 colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
#                 axes=ax)
#             ax.dist = 7
        
#         i += 1
        
# for label, id_grid in zip(grps_label, [0, 2]):
#     ax = fig.add_subplot(grid[id_grid])
#     ax.axis('off')
#     ax.text(1, 0.9, label, ha='center', fontdict={'fontsize':40})
    
# cax = plt.axes([1.05, 0.32, 0.03, 0.4])
# cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
# cbar.set_ticks([])
# cbar.ax.set_title('max', fontdict={'fontsize':30}, pad=20)
# cbar.ax.set_xlabel('min', fontdict={'fontsize':30}, labelpad=20)

# plt.show()