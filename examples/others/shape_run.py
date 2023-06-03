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

from neuroshape.nipype.interfaces.workbench.metric import MetricResample, MetricGradient
import os
from numpy import inf
from neuroshape.eigenmaps import LBO
from neuroshape.permutation import Permutation
from neuroshape.utils.dataio import dataio, save, run, fname_presuffix, split_filename
from pathlib import Path
import os.path as op

#get current environment
env = os.environ

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
grps = ['HC','P']
grps_label = ['Healthy Cohort (HC)', 'Early Psychosis (EP)']

i = 0
Vn = 2

for grp in grps:
    print(f'Compute {grp}-average LBO eigenmaps and magnitudes\n')
    for hemi in ['L','R']:
        #load in files
        text_file = f'{baseFolder}/{grp_folder}/{grp}/subjects.txt'
        subjects = np.loadtxt(text_file, dtype='str').squeeze()
        hemi_files = [f'{baseFolder}/{grp_folder}/{grp}/{subject}.{hemi}.{surf}.native.surf.gii' for subject in subjects]
        hemi_data = [nib.load(file).agg_data() for file in hemi_files]
        
        #initialize Shape object
        shape = LBO(Vn=Vn, n_jobs=20)
        
        #compute eigenmaps
        evs = shape(hemi_data)
        
        resampled_files = []
        
        for ii in range(len(hemi_data)):
            pth, fname, ext = split_filename(hemi_files[ii])
            out_file = op.join(pth,
                               f'ev{Vn}.' +
                               fname[:-len(type_surface)] +
                               type_metric +
                               ext)
            
            #if not Path(out_file).exists():
                #save file
            out = save(evs[ii], out_file)
            
            resampled_file = op.join(out_file[:2+len(hemi)-len(surf)-len(type_metric)-len(ext)] + 
                                     f'.{common_space_name}' +
                                     f'.{type_metric}' +
                                     ext)
            
            #if not Path(resampled_file).exists():
                #resample file
            print('Resampling metric file...')
            metrics = MetricResample()
            metrics.inputs.in_file = out_file
            metrics.inputs.method = 'BARYCENTRIC'
            metrics.inputs.current_sphere = f'{baseFolder}/{grp_folder}/{grp}/{subjects[ii]}.{hemi}.sphere.native.surf.gii'
            metrics.inputs.new_sphere = f'{baseFolder}/{grp_folder}/{grp}/{subjects[ii]}.{hemi}.sphere.{common_space_name}.surf.gii'
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
        
    #bookkeeping    
    if grp == 'HC':
        HC_ev_all_lh = evs_all_lh
        HC_ev_all_rh = evs_all_rh
        
    else:
        P_ev_all_lh = evs_all_lh
        P_ev_all_rh = evs_all_rh
        
HC_ev_all = np.hstack((HC_ev_all_lh, HC_ev_all_rh))
P_ev_all = np.hstack((P_ev_all_lh, P_ev_all_rh))

# =============================================================================
# #### Permutation ####
# =============================================================================

#initialize Permutation object
n_perms = 1000
perm = Permutation(HC_ev_all, P_ev_all, seed=20, n_jobs=10)

surrs_all = perm(n_perms)

#calculate z-scores
data_ref = np.mean(HC_ev_all, axis=0) - np.mean(P_ev_all, axis=0)
data_ref_inv = np.mean(P_ev_all, axis=0) - np.mean(HC_ev_all, axis=0)

data_all = np.mean(surrs_all[0], axis=1).squeeze() - np.mean(surrs_all[1], axis=1).squeeze()
data_all_inv = np.mean(surrs_all[1], axis=1).squeeze() - np.mean(surrs_all[0], axis=1).squeeze()

mean_data = np.mean(data_all, axis=0)
mean_data_inv = np.mean(data_all_inv, axis=0)

std_all = np.std(data_all, axis=0)
std_all_inv = np.std(data_all_inv, axis=0)

zscore = np.divide((data_ref - mean_data), std_all)
zscore_inv = np.divide((data_ref_inv - mean_data_inv), std_all_inv)

#save images

data_zscore_lh = nib.GiftiImage()
data_zscore_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore[:len(zscore)//2]))
data_zscore_rh = nib.GiftiImage()
data_zscore_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore[len(zscore)//2:]))

nib.save(data_zscore_lh, f'{baseFolder}/{grp_folder}/ev{Vn}.L.HC-P.zscore.{n_perms}_perms.32k_fs_LR.func.gii')
nib.save(data_zscore_rh, f'{baseFolder}/{grp_folder}/ev{Vn}.R.HC-P.zscore.{n_perms}_perms.32k_fs_LR.func.gii')

#inverse
data_zscore_lh_inv = nib.GiftiImage()
data_zscore_lh_inv.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore_inv[:len(zscore_inv)//2]))
data_zscore_rh_inv = nib.GiftiImage()
data_zscore_rh_inv.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore_inv[len(zscore_inv)//2:]))

nib.save(data_zscore_lh_inv, f'{baseFolder}/{grp_folder}/L.Vn{Vn}.P-HC.zscore.32k_fs_LR.func.gii')
nib.save(data_zscore_rh_inv, f'{baseFolder}/{grp_folder}/R.Vn{Vn}.P-HC.zscore.32k_fs_LR.func.gii')

#calculate p-values
pval_sup = np.sum([(data_ref >= i) for i in data_all], axis=0)/n_perms

data_pval_sup_lh = nib.GiftiImage()
data_pval_sup_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pval_sup[:len(zscore)//2].astype('float32')))
data_pval_sup_rh = nib.GiftiImage()
data_pval_sup_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pval_sup[len(zscore)//2:].astype('float32')))

nib.save(data_pval_sup_lh, f'{baseFolder}/{grp_folder}/ev{Vn}.L.HC-P.pval.{n_perms}_perms.32k_fs_LR.func.gii')
nib.save(data_pval_sup_rh, f'{baseFolder}/{grp_folder}/ev{Vn}.R.HC-P.pval.{n_perms}_perms.32k_fs_LR.func.gii')

#inverse
pval_inf = np.sum([(data_ref <= i) for i in data_all], axis=0)/n_perms

data_pval_inf_lh = nib.GiftiImage()
data_pval_inf_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pval_sup[:len(zscore)//2].astype('float32')))
data_pval_inf_rh = nib.GiftiImage()
data_pval_inf_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pval_sup[len(zscore)//2:].astype('float32')))

nib.save(data_pval_inf_lh, f'{baseFolder}/{grp_folder}/ev{Vn}.L.P-HC.pval.{n_perms}_perms.32k_fs_LR.func.gii')
nib.save(data_pval_inf_rh, f'{baseFolder}/{grp_folder}/ev{Vn}.R.P-HC.pval.{n_perms}_perms.32k_fs_LR.func.gii')

#threshold z-scores by pvals

zscore_thr = np.where(pval_inf >= 0.95, zscore, 0.) + np.where(pval_sup >= 0.95, zscore, 0.)

data_zscore_thr_lh = nib.GiftiImage()
data_zscore_thr_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore_thr[:len(zscore_thr)//2].astype('float32')))
data_zscore_thr_rh = nib.GiftiImage()
data_zscore_thr_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore_thr[len(zscore_thr)//2:].astype('float32')))

nib.save(data_zscore_thr_lh, f'{baseFolder}/{grp_folder}/ev{Vn}.L.HC-P.zscore_thr.{n_perms}_perms.32k_fs_LR.func.gii')
nib.save(data_zscore_thr_rh, f'{baseFolder}/{grp_folder}/ev{Vn}.R.HC-P.zscore_thr.{n_perms}_perms.32k_fs_LR.func.gii')       

#save group averages for gradient mapping
data_HC_ev_lh = nib.GiftiImage()
data_HC_ev_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(np.mean(HC_ev_all_lh, axis=0).astype('float32')))

data_HC_ev_rh = nib.GiftiImage()
data_HC_ev_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(np.mean(HC_ev_all_rh, axis=0).astype('float32')))

nib.save(data_HC_ev_lh, f'{baseFolder}/{grp_folder}/HC.ev{Vn}.L.32k_fs_LR.func.gii')
nib.save(data_HC_ev_rh, f'{baseFolder}/{grp_folder}/HC.ev{Vn}.R.32k_fs_LR.func.gii')

#P
data_P_ev_lh = nib.GiftiImage()
data_P_ev_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(np.mean(P_ev_all_lh, axis=0).astype('float32')))

data_P_ev_rh = nib.GiftiImage()
data_P_ev_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(np.mean(P_ev_all_rh, axis=0).astype('float32')))

nib.save(data_P_ev_lh, f'{baseFolder}/{grp_folder}/P.ev{Vn}.L.32k_fs_LR.func.gii')
nib.save(data_P_ev_rh, f'{baseFolder}/{grp_folder}/P.ev{Vn}.R.32k_fs_LR.func.gii')

# =============================================================================
# #### With normalization ####
# =============================================================================

norm_factor = np.max(np.mean(np.vstack((HC_ev_all, P_ev_all)), axis=0)) - np.min(np.min(np.vstack((HC_ev_all, P_ev_all)), axis=0))
HC_ev_all_norm = HC_ev_all / norm_factor
P_ev_all_norm = P_ev_all / norm_factor

#perform permutation
n_perms = 1000
perm = Permutation(HC_ev_all_norm, P_ev_all_norm, seed=20, n_jobs=10)

surrs_all = perm(n_perms)

#calculate z-scores
data_ref_norm = np.mean(HC_ev_all_norm, axis=0) - np.mean(P_ev_all_norm, axis=0)
data_ref_inv_norm = np.mean(P_ev_all_norm, axis=0) - np.mean(HC_ev_all_norm, axis=0)

data_all_norm = np.mean(surrs_all[0], axis=1).squeeze() - np.mean(surrs_all[1], axis=1).squeeze()
data_all_inv_norm = np.mean(surrs_all[1], axis=1).squeeze() - np.mean(surrs_all[0], axis=1).squeeze()

mean_data_norm = np.mean(data_all_norm, axis=0)
mean_data_inv_norm = np.mean(data_all_inv_norm, axis=0)

std_all_norm = np.std(data_all_norm, axis=0)
std_all_inv_norm = np.std(data_all_inv_norm, axis=0)

zscore_norm = np.divide((data_ref_norm - mean_data_norm), std_all_norm)
zscore_inv_norm = np.divide((data_ref_inv_norm - mean_data_inv_norm), std_all_inv_norm)

#save images

data_zscore_norm_lh = nib.GiftiImage()
data_zscore_norm_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore_norm[:len(zscore_norm)//2]))
data_zscore_norm_rh = nib.GiftiImage()
data_zscore_norm_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore_norm[len(zscore_norm)//2:]))

nib.save(data_zscore_norm_lh, f'{baseFolder}/{grp_folder}/ev{Vn}.L.HC-P.zscore.norm.{n_perms}_perms.32k_fs_LR.func.gii')
nib.save(data_zscore_norm_rh, f'{baseFolder}/{grp_folder}/ev{Vn}.R.HC-P.zscore.norm.{n_perms}_perms.32k_fs_LR.func.gii')

#inverse
data_zscore_norm_lh_inv = nib.GiftiImage()
data_zscore_norm_lh_inv.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore_inv_norm[:len(zscore_inv_norm)//2]))
data_zscore_norm_rh_inv = nib.GiftiImage()
data_zscore_norm_rh_inv.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore_inv_norm[len(zscore_inv_norm)//2:]))

nib.save(data_zscore_norm_lh_inv, f'{baseFolder}/{grp_folder}/L.Vn{Vn}.P-HC.zscore.norm.{n_perms}_perms.32k_fs_LR.func.gii')
nib.save(data_zscore_norm_rh_inv, f'{baseFolder}/{grp_folder}/R.Vn{Vn}.P-HC.zscore.norm.{n_perms}_perms.32k_fs_LR.func.gii')

#calculate p-values
pval_sup_norm = np.sum([(data_ref_norm >= i) for i in data_all_norm], axis=0)/n_perms

data_pval_sup_norm_lh = nib.GiftiImage()
data_pval_sup_norm_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pval_sup_norm[:len(zscore_norm)//2].astype('float32')))
data_pval_sup_norm_rh = nib.GiftiImage()
data_pval_sup_norm_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pval_sup_norm[len(zscore_norm)//2:].astype('float32')))

nib.save(data_pval_sup_norm_lh, f'{baseFolder}/{grp_folder}/ev{Vn}.L.HC-P.pval.norm.{n_perms}_perms.32k_fs_LR.func.gii')
nib.save(data_pval_sup_norm_rh, f'{baseFolder}/{grp_folder}/ev{Vn}.R.HC-P.pval.norm.{n_perms}_perms.32k_fs_LR.func.gii')

#inverse
pval_inf_norm = np.sum([(data_ref_norm <= i) for i in data_all_norm], axis=0)/n_perms

data_pval_inf_norm_lh = nib.GiftiImage()
data_pval_inf_norm_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pval_inf_norm[:len(zscore)//2].astype('float32')))
data_pval_inf_norm_rh = nib.GiftiImage()
data_pval_inf_norm_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(pval_inf_norm[len(zscore)//2:].astype('float32')))

nib.save(data_pval_inf_norm_lh, f'{baseFolder}/{grp_folder}/ev{Vn}.L.P-HC.pval.norm.{n_perms}_perms.32k_fs_LR.func.gii')
nib.save(data_pval_inf_norm_rh, f'{baseFolder}/{grp_folder}/ev{Vn}.R.P-HC.pval.norm.{n_perms}_perms.32k_fs_LR.func.gii')

#threshold z-scores by pvals

zscore_thr_norm = np.where(pval_inf_norm >= 0.95, zscore_norm, 0.) + np.where(pval_sup_norm >= 0.95, zscore_norm, 0.)

data_zscore_thr_norm_lh = nib.GiftiImage()
data_zscore_thr_norm_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore_thr_norm[:len(zscore_thr_norm)//2].astype('float32')))
data_zscore_thr_norm_rh = nib.GiftiImage()
data_zscore_thr_norm_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(zscore_thr_norm[len(zscore_thr_norm)//2:].astype('float32')))

nib.save(data_zscore_thr_norm_lh, f'{baseFolder}/{grp_folder}/ev{Vn}.L.HC-P.zscore_thr.norm.{n_perms}_perms.32k_fs_LR.func.gii')
nib.save(data_zscore_thr_norm_rh, f'{baseFolder}/{grp_folder}/ev{Vn}.R.HC-P.zscore_thr.norm.{n_perms}_perms.32k_fs_LR.func.gii')       

#save group averages for gradient mapping
HC_ev_mean_norm_lh = np.mean(HC_ev_all_norm, axis=0)[:HC_ev_all_norm.shape[1]//2].astype('float32')
HC_ev_mean_norm_rh = np.mean(HC_ev_all_norm, axis=0)[HC_ev_all_norm.shape[1]//2:].astype('float32')

data_HC_ev_norm_lh = nib.GiftiImage()
data_HC_ev_norm_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(HC_ev_mean_norm_lh))

data_HC_ev_norm_rh = nib.GiftiImage()
data_HC_ev_norm_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(HC_ev_mean_norm_rh))

nib.save(data_HC_ev_norm_lh, f'{baseFolder}/{grp_folder}/ev{Vn}.HC.norm.L.32k_fs_LR.func.gii')
nib.save(data_HC_ev_norm_rh, f'{baseFolder}/{grp_folder}/ev{Vn}.HC.norm.R.32k_fs_LR.func.gii')

#P
P_ev_mean_norm_lh = np.mean(P_ev_all_norm, axis=0)[:P_ev_all_norm.shape[1]//2].astype('float32')
P_ev_mean_norm_rh = np.mean(P_ev_all_norm, axis=0)[P_ev_all_norm.shape[1]//2:].astype('float32')

data_HC_ev_norm_lh = nib.GiftiImage()
data_HC_ev_norm_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(P_ev_mean_norm_lh))

data_HC_ev_norm_rh = nib.GiftiImage()
data_HC_ev_norm_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(P_ev_mean_norm_rh))

nib.save(data_HC_ev_norm_lh, f'{baseFolder}/{grp_folder}/ev{Vn}.P.norm.L.32k_fs_LR.func.gii')
nib.save(data_HC_ev_norm_rh, f'{baseFolder}/{grp_folder}/ev{Vn}.P.norm.R.32k_fs_LR.func.gii')


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