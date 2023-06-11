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
import os.path as op

from neuroshape.nipype.interfaces.workbench.metric import MetricResample, MetricGradient
import os
from numpy import inf
from neuroshape.poly_eigenmaps import PolyLBO
from neuroshape.permutation import Permutation
from neuroshape.utils.dataio import dataio, save, run, fname_presuffix, split_filename
from pathlib import Path
import os.path as op
from neuromaps.datasets.atlases import fetch_atlas
from neuromaps.stats import compare_images, permtest_metric

# =============================================================================
# #### Path sourcing ####
# =============================================================================

codeFolder = '/Volumes/Scratch/functional_integration_psychosis/code/subcortex'
resultFolder = f'{codeFolder}/result/HCP-EP-striatum'
meshFolder = f'{codeFolder}/meshes'
surfFolder = '/Volumes/Scratch/functional_integration_psychosis/preprocessed/HCP-EP/LBO/surfaces'

baseFolder = "/Volumes/Scratch/functional_integration_psychosis/preprocessed/HCP-EP/LBO"
dataFolder = "/Volumes/Scratch/functional_integration_psychosis/code/neuroshape/data"
#fsFolder = "/Volumes/Scratch/functional_integration_psychosis/preprocessing/HCP-EP/FS/fmriresults01"

# =============================================================================
# #### Load precomputed arrays ####
# =============================================================================

HC_ev_mean = np.load(f'{dataFolder}/HCP-EP_HC_ev_mean.npy')
P_ev_mean = np.load(f'{dataFolder}/HCP-EP_EP_ev_mean.npy')
HC_ev_subwise_norm = np.load(f'{dataFolder}/HCP-EP_HC_ev_subwise_norm.npy')
P_ev_subwise_norm = np.load(f'{dataFolder}/HCP-EP_P_ev_subwise_norm.npy')

HC_ev_mean_plot = np.zeros(HC_ev_mean.shape)
P_ev_mean_plot = np.zeros(P_ev_mean.shape)

# TODO standardize arrays for comparison



# =============================================================================
# #### Set up figure for LBO plots ####
# =============================================================================

# create figure
fig = plt.figure(figsize=(20,9), constrained_layout=False)
grid = gridspec.GridSpec(
    2, 5, left=0., right=1., bottom=0., top=1.,
    height_ratios=[1.,1.], width_ratios=[1,1,1,1,1],
    hspace=0.0, wspace=0.0)

task = 'resting-state'
#img = image.load_img(f'{codeFolder}/masks/subcortex_mask.nii')
stri_file = f'{codeFolder}/masks/striatum_cropped.nii'
stri_msk = image.load_img(stri_file)
#sub_msk = image.load_img(f'{codeFolder}/masks/subcortex_mask.nii')
fslr = fetch_atlas('fsLR', '32k')

hemi = 'left'
texture = HC_ev_subwise_norm[0] #change to fit purpose
cohort = 'HCP-EP'
task = 'Resting-state'
#view = 'lateral'

grps = ['HC','P']
grps_label = ['Healthy Cohort (HC)', 'Early Psychosis (EP)']
view_label = ['Lateral view', 'Medial view']


for Vn in [0, 1, 2, 3]:
    vmin, vmax = np.min(texture[Vn]), np.max(texture[Vn])
    if hemi == 'left':
        for col in [Vn, Vn+5]:
            if col == Vn:
                view = 'lateral'
            else:
                view = 'medial'
            infl_mesh = fslr.inflated[0]
            ax = fig.add_subplot(grid[col], projection='3d')
            plotting.plot_surf(
                infl_mesh, texture[Vn,:texture.shape[1]//2], hemi=hemi, view=view,
                colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
                axes=ax)
            ax.dist = 7
                
    else:
        for col in [Vn, Vn+5]:
            if col == Vn:
                view = 'lateral'
            else:
                view = 'medial'
            infl_mesh = fslr.inflated[1]
            ax = fig.add_subplot(grid[col], projection='3d')
            plotting.plot_surf(
                infl_mesh, texture[Vn,texture.shape[1]//2:], hemi=hemi, view=view,
                colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
                axes=ax)
            ax.dist = 7
            
Vn = 199
vmin, vmax = np.min(texture[Vn]), np.max(texture[Vn])
for col in [4, 9]:
    if col == 4:
        view = 'lateral'
    else:
        view = 'medial'
    if hemi == 'left':
        infl_mesh = fslr.inflated[0]
        ax = fig.add_subplot(grid[col], projection='3d')
        plotting.plot_surf(
            infl_mesh, texture[Vn,:texture.shape[1]//2], hemi=hemi, view=view,
            colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
            axes=ax)
        ax.dist = 7
                    
    else:
        infl_mesh = fslr.inflated[1]
        ax = fig.add_subplot(grid[col], projection='3d')
        plotting.plot_surf(
            infl_mesh, texture[Vn,texture.shape[1]//2:], hemi=hemi, view=view,
            colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
            axes=ax)
        ax.dist = 7    
    
for id_grid in [6]:
    ax = fig.add_subplot(grid[2])
    ax.axis('off')
    ax.text(0.5, 0.9, view_label[0], ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid-1])
    ax.axis('off')
    ax.text(0.5, 0, 'Mode 1', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')
    ax.text(0.5, 0, 'Mode 2', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid+1])
    ax.axis('off')
    ax.text(0.5, 0, 'Mode 3', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid+2])
    ax.axis('off')
    ax.text(0.5, 0, 'Mode 4', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[7])
    ax.axis('off')
    ax.text(0.5, 0.9, view_label[1], ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid+3])
    ax.axis('off')
    ax.text(0.5, 0, 'Mode N', ha='center', fontdict={'fontsize':30})
 
# colorbar
cax = plt.axes([1.05, 0.32, 0.03, 0.4])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.set_ticks([])
cbar.ax.set_title('max', fontdict={'fontsize':30}, pad=20)
cbar.ax.set_xlabel('min', fontdict={'fontsize':30}, labelpad=20)

plt.show()

# =============================================================================
# #### Functional eigenmodes ####
# =============================================================================

codeFolder = '/Volumes/Scratch/functional_integration_psychosis/code/subcortex'
resultFolder = f'{codeFolder}/result/HCP-EP-striatum'

Vn = 2
grp = 'HC'
task = 'resting-state'

img_file = f'{resultFolder}/tasks/{task}/cohorts/{grp}/Vn{Vn}_eigenvector_projection_striatum_2mm.nii'
img = image.load_img(img_file)

from neuromaps import transforms
func_surf_fslr = transforms.mni152_to_fslr(img, '32k')

texture_func = np.hstack((func_surf_fslr[0].agg_data(), func_surf_fslr[1].agg_data()))

texture_func_norm = texture_func/np.linalg.norm(texture_func, axis=0)
texture_func_norm = texture_func_norm.reshape(-1,1)

# =============================================================================
# #### Show reconstruction and plot side-by-side with empirical ####
# =============================================================================

# change cmap
cmap = plt.get_cmap('viridis')

# load betas and empirical functional Vn
grp = 'HC'
Vn = 2
betas = np.load(f'{dataFolder}/HCP-EP_{grp}_betas.npy')
phi = np.load(f'{dataFolder}/HCP-EP_{grp}_func_Vn{Vn}.npy').reshape(-1,)

# calc recon
recon = np.vstack(np.matmul(betas[sub].T, HC_ev_subwise[sub]).reshape(-1,) for sub in range(betas.shape[0]))

# normalize recon
recon_norm = np.vstack(recon[idx]/np.linalg.norm(recon[idx], ord=np.inf) for idx in range(recon.shape[0]))

# compare images
corr = np.vstack(compare_images(recon_norm[idx], phi, metric='pearsonr') for idx in range(recon_norm.shape[0]))

average_corr = np.mean(corr, axis=0)

# create figure
fig = plt.figure(figsize=(15,9), constrained_layout=False)
grid = gridspec.GridSpec(
    2, 2, left=0., right=1., bottom=0., top=1.,
    height_ratios=[1.,1.], width_ratios=[1,1],
    hspace=0.0, wspace=0.0)

hemi = 'left'
vmin, vmax = np.min(phi), np.max(phi)

for idx in [0, 1]:
    if idx == 0:
        texture = phi
        for col in [0, 1]:
            if col == 1:
                view = 'lateral'
            else:
                view = 'medial'
            if hemi == 'left':
                ax = fig.add_subplot(grid[col], projection='3d')
                plotting.plot_surf(
                    infl_mesh, texture[:len(texture)//2], hemi=hemi, view=view,
                    colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
                    axes=ax)
                ax.dist = 7
            else:
                ax = fig.add_subplot(grid[col], projection='3d')
                plotting.plot_surf(
                    infl_mesh, texture[len(texture)//2:], hemi=hemi, view=view,
                    colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
                    axes=ax)
                ax.dist = 7
                
    else:
        texture = np.mean(recon_norm, axis=0)
        for col in [0, 1]:
            if col == 1:
                view = 'lateral'
            else:
                view = 'medial'
            if hemi == 'left':
                ax = fig.add_subplot(grid[col+2], projection='3d')
                plotting.plot_surf(
                    infl_mesh, texture[:len(texture)//2], hemi=hemi, 
                    view=view, colorbar=False, cmap=cmap, vmin=vmin, 
                    vmax=vmax, axes=ax)
                ax.dist = 7
            else:
                ax = fig.add_subplot(grid[col+2], projection='3d')
                plotting.plot_surf(
                    infl_mesh, texture[len(texture)//2:], hemi=hemi, 
                    view=view, colorbar=False, cmap=cmap, vmin=vmin, 
                    vmax=vmax, axes=ax)
                ax.dist = 7

for label, id_grid in zip([fr'$\phi_{Vn-1}$ : {task} - {grp}',
                           fr'Reconstruction of $\phi_{Vn-1}$,'
                           'r = {average_corr:.3f}'], [0, 2]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')
    ax.text(1, 0.9, label, ha='center', fontdict={'fontsize':30})
    
# colorbar
cax = plt.axes([1.05, 0.32, 0.03, 0.4])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.set_ticks([])
cbar.ax.set_title('max', fontdict={'fontsize':30}, pad=20)
cbar.ax.set_xlabel('min', fontdict={'fontsize':30}, labelpad=20)

plt.show()

# =============================================================================
# #### Hemisphere-wise model ####
# =============================================================================

psi_lh = HC_ev_subwise[:,:,:HC_ev_subwise.shape[2]//2]
phi_lh = texture_func_norm[:len(texture_func_norm)//2]

betas_lh = np.vstack(np.matmul(psi_subj, phi_lh) for psi_subj in psi_lh).reshape(len(psi_lh),-1)

psi_rh = HC_ev_subwise[:,:,HC_ev_subwise.shape[2]//2:]
phi_rh = texture_func_norm[len(texture_func_norm)//2:]

betas_rh = np.vstack(np.matmul(psi_subj, phi_rh) for psi_subj in psi_rh).reshape(len(psi_rh),-1)


# =============================================================================
# #### Show reconstruction and plot side-by-side with empirical ####
# =============================================================================

# load betas and empirical functional Vn
grp = 'HC'
Vn = 2
#betas = np.load(f'{dataFolder}/HCP-EP_{grp}_betas.npy')
#phi = np.load(f'{dataFolder}/HCP-EP_{grp}_func_Vn{Vn}.npy').reshape(-1,)

# calc recon
recon_lh = np.vstack(np.matmul(betas_lh[sub].T, psi_lh[sub]).reshape(-1,) for sub in range(betas_lh.shape[0]))
recon_rh = np.vstack(np.matmul(betas_rh[sub].T, psi_rh[sub]).reshape(-1,) for sub in range(betas_rh.shape[0]))

# normalize recon
recon_norm_lh = np.vstack(recon_lh[idx]/np.linalg.norm(recon_lh[idx], ord=np.inf) for idx in range(recon_lh.shape[0]))
recon_norm_rh = np.vstack(recon_rh[idx]/np.linalg.norm(recon_rh[idx], ord=np.inf) for idx in range(recon_rh.shape[0]))

# compare images
corr_lh = np.vstack(compare_images(recon_norm_lh[idx], phi_lh, metric='spearmanr') for idx in range(recon_norm_lh.shape[0]))
corr_rh = np.vstack(compare_images(recon_norm_rh[idx], phi_rh, metric='spearmanr') for idx in range(recon_norm_rh.shape[0]))

average_corr_hemiwise = np.mean((np.mean(corr_lh, axis=0), np.mean(corr_rh, axis=0)))

# create figure
fig = plt.figure(figsize=(15,9), constrained_layout=False)
grid = gridspec.GridSpec(
    2, 2, left=0., right=.8, bottom=0., top=1.,
    height_ratios=[1.,1.], width_ratios=[1,1],
    hspace=0.0, wspace=0.0)

hemi = 'left'
vmin, vmax = np.min(phi), np.max(phi)

for idx in [0, 1]:
    if idx == 0:
        texture = phi_lh
        for col in [0, 1]:
            if col == 1:
                view = 'lateral'
            else:
                view = 'medial'
            if hemi == 'left':
                ax = fig.add_subplot(grid[col], projection='3d')
                plotting.plot_surf(
                    infl_mesh, texture, hemi=hemi, view=view,
                    colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
                    axes=ax)
                ax.dist = 7
            else:
                ax = fig.add_subplot(grid[col], projection='3d')
                plotting.plot_surf(
                    infl_mesh, texture[len(texture)//2:], hemi=hemi, view=view,
                    colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
                    axes=ax)
                ax.dist = 7
                
    else:
        texture = np.mean(recon_norm_lh, axis=0)
        for col in [0, 1]:
            if col == 1:
                view = 'lateral'
            else:
                view = 'medial'
            if hemi == 'left':
                ax = fig.add_subplot(grid[col+2], projection='3d')
                plotting.plot_surf(
                    infl_mesh, texture, hemi=hemi, 
                    view=view, colorbar=False, cmap=cmap, vmin=vmin, 
                    vmax=vmax, axes=ax)
                ax.dist = 7
            else:
                ax = fig.add_subplot(grid[col+2], projection='3d')
                plotting.plot_surf(
                    infl_mesh, texture, hemi=hemi, 
                    view=view, colorbar=False, cmap=cmap, vmin=vmin, 
                    vmax=vmax, axes=ax)
                ax.dist = 7

for label, id_grid in zip([fr'$\phi_{Vn-1}$ : {task} - {grp}', fr'Reconstruction of $\phi_{Vn-1}$, r = {average_corr_hemiwise:.3f}'], [0, 2]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')
    ax.text(1, 0.9, label, ha='center', fontdict={'fontsize':30})
    
# colorbar
cax = plt.axes([0.8, 0.32, 0.03, 0.4])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.set_ticks([])
cbar.ax.set_title('max', fontdict={'fontsize':30}, pad=20)
cbar.ax.set_xlabel('min', fontdict={'fontsize':30}, labelpad=20)

plt.show()


# =============================================================================
# #### Subject-wise reconstruction ####
# =============================================================================

# TODO
