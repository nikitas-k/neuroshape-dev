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
from neuromaps.stats import compare_images, permtest_metric
from lapy import TriaMesh, TetMesh, TetIO
from lapy.ShapeDNA import compute_shapedna
from neuromaps import transforms
from neuromaps import nulls
from neuroshape.utils.eigen import maximise_recon_metric
from sklearn.preprocessing import normalize
from neuroshape.utils.recon import recon_parallel
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from neuroshape.utils.zscore_avg_method import zscore_avg_method
from neuroshape.utils.compare_parallel import compare_geomodes_parallel

cmap = plt.get_cmap('viridis')
fslr = fetch_atlas('fsLR', density='32k')

# =============================================================================
# #### Path sourcing ####
# =============================================================================

codeFolder = '/Volumes/Scratch/functional_integration_psychosis/code/subcortex'
resultFolder = f'{codeFolder}/result/HCP-EP-striatum'
meshFolder = f'{codeFolder}/meshes'
surfFolder = '/Volumes/Scratch/functional_integration_psychosis/preprocessed/HCP-EP/LBO/surfaces'

baseFolder = "/Volumes/Scratch/functional_integration_psychosis/preprocessed/HCP-EP/LBO"

dataFolder = "/Volumes/Scratch/functional_integration_psychosis/code/neuroshape/data"

HC_betas_subwise_lh = np.load(f'{dataFolder}/HCP-EP_HC_Vn2_betas_subwise_lh.npy')
HC_betas_subwise_rh = np.load(f'{dataFolder}/HCP-EP_HC_Vn2_betas_subwise_rh.npy')

P_betas_subwise_lh = np.load(f'{dataFolder}/HCP-EP_EP_Vn2_betas_subwise_lh.npy')
P_betas_subwise_rh = np.load(f'{dataFolder}/HCP-EP_EP_Vn2_betas_subwise_rh.npy')

# =============================================================================
# #### Geometric eigenmodes ####
# =============================================================================

# load data
dataFolder = "/Volumes/Scratch/functional_integration_psychosis/code/neuroshape/data"
baseFolder = "/Volumes/Scratch/functional_integration_psychosis/preprocessed/HCP-EP/LBO"

HC_ev_mean = np.load(f'{dataFolder}/HCP-EP_HC_ev_mean.npy')
P_ev_mean = np.load(f'{dataFolder}/HCP-EP_EP_ev_mean.npy')
HC_ev_subwise_norm = np.load(f'{dataFolder}/HCP-EP_HC_ev_subwise_norm.npy')
P_ev_subwise_norm = np.load(f'{dataFolder}/HCP-EP_P_ev_subwise_norm.npy')

# rename for readbility (psi is already transposed)

HC_Psi = HC_ev_subwise_norm
HC_Psi_lh = HC_Psi[:,:,:HC_Psi.shape[2]//2]
HC_Psi_rh = HC_Psi[:,:,HC_Psi.shape[2]//2:]

#renormalize
for sub in range(HC_Psi.shape[0]):
    for j in range(HC_Psi.shape[1]):
        HC_Psi_lh[sub,j] = (HC_Psi_lh[sub,j]-HC_Psi_lh[sub,j].mean())/(HC_Psi_lh[sub,j].std())
        HC_Psi_rh[sub,j] = (HC_Psi_rh[sub,j]-HC_Psi_rh[sub,j].mean())/(HC_Psi_rh[sub,j].std())
        
P_Psi = P_ev_subwise_norm
P_Psi_lh = P_Psi[:,:,:P_Psi.shape[2]//2]
P_Psi_rh = P_Psi[:,:,P_Psi.shape[2]//2:]

#renormalize
for sub in range(P_Psi.shape[0]):
    for j in range(P_Psi.shape[1]):
        P_Psi_lh[sub,j] = (P_Psi_lh[sub,j]-P_Psi_lh[sub,j].mean())/(P_Psi_lh[sub,j].std())
        P_Psi_rh[sub,j] = (P_Psi_rh[sub,j]-P_Psi_rh[sub,j].mean())/(P_Psi_rh[sub,j].std())

# =============================================================================
# #### HC intersubject variability ####
# =============================================================================

corr_lh = compare_geomodes_parallel(np.abs(HC_Psi_lh), n_jobs=20)
corr_rh = compare_geomodes_parallel(np.abs(HC_Psi_rh), n_jobs=20)

average_corr_lh = np.mean(corr_lh, axis=0)
average_corr_rh = np.mean(corr_rh, axis=0)

std_corr_lh = np.std(corr_lh, axis=0)
std_corr_rh = np.std(corr_rh, axis=0)

j = len(average_corr_lh+1)
fig = plt.figure(figsize=(10,8), constrained_layout=False)
ci = 1.96*std_corr_lh/np.sqrt(j+1)
ax = fig.add_subplot()
x = range(1,j+1)
y = average_corr_lh
ax.plot(x, y, linewidth=2, color='b', label='Left hemisphere')
ax.fill_between(x, (y-ci), (y+ci), color='b', alpha=.2)
y = average_corr_rh
ci = 1.96*std_corr_rh/np.sqrt(j+1)
ax.plot(x, y, linewidth=2, color='r', label='Right hemisphere')
ax.fill_between(x, (y-ci), (y+ci), color='r', alpha=.2)
plt.ylabel('Correlation', fontdict={'fontsize':20})
plt.xlabel(r'Mode $\psi$', fontdict={'fontsize':20})
plt.title('Healthy control correlation across modes', fontdict={'fontsize':20})
plt.xlim([1, 200])
plt.ylim([0, 1.0])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=15)
plt.show()

# plot a few examples to show

j = 175
texture = HC_Psi_lh[0,j]
view = 'lateral'
vmin = np.min(texture)
vmax = np.max(texture)
hemi = 'left'
cmap = plt.get_cmap('coolwarm')

fig = plt.figure(figsize=(15,9), constrained_layout=False)
pial_mesh = 'data/fsaverage.L.pial_orig.32k_fs_LR.surf.gii'

ax = fig.add_subplot(projection='3d')
plotting.plot_surf(
    pial_mesh, texture, hemi=hemi, 
    view=view, colorbar=False, cmap=cmap, vmin=vmin, 
    vmax=vmax, axes=ax)
ax.dist = 7

plt.show()

texture = HC_Psi_lh[6,j]
view = 'lateral'
vmin = np.min(texture)
vmax = np.max(texture)
hemi = 'left'
cmap = plt.get_cmap('coolwarm')

fig = plt.figure(figsize=(15,9), constrained_layout=False)
pial_mesh = 'data/fsaverage.L.pial_orig.32k_fs_LR.surf.gii'

ax = fig.add_subplot(projection='3d')
plotting.plot_surf(
    pial_mesh, texture, hemi=hemi, 
    view=view, colorbar=False, cmap=cmap, vmin=vmin, 
    vmax=vmax, axes=ax)
ax.dist = 7

plt.show()

texture = HC_Psi_lh[22,j]
view = 'lateral'
vmin = np.min(texture)
vmax = np.max(texture)
hemi = 'left'
cmap = plt.get_cmap('coolwarm')

fig = plt.figure(figsize=(15,9), constrained_layout=False)
pial_mesh = 'data/fsaverage.L.pial_orig.32k_fs_LR.surf.gii'

ax = fig.add_subplot(projection='3d')
plotting.plot_surf(
    pial_mesh, texture, hemi=hemi, 
    view=view, colorbar=False, cmap=cmap, vmin=vmin, 
    vmax=vmax, axes=ax)
ax.dist = 7

plt.show()

# =============================================================================
# #### P intersubject variability ####
# =============================================================================

corr_lh = compare_geomodes_parallel(np.abs(P_Psi_lh), n_jobs=20)
corr_rh = compare_geomodes_parallel(np.abs(P_Psi_rh), n_jobs=20)

average_corr_lh = np.mean(corr_lh, axis=0)
average_corr_rh = np.mean(corr_rh, axis=0)

std_corr_lh = np.std(corr_lh, axis=0)
std_corr_rh = np.std(corr_rh, axis=0)

j = len(average_corr_lh+1)
fig = plt.figure(figsize=(10,8), constrained_layout=False)
ci = 1.96*std_corr_lh/np.sqrt(j+1)
ax = fig.add_subplot()
x = range(1,j+1)
y = average_corr_lh
ax.plot(x, y, linewidth=2, color='b', label='Left hemisphere')
ax.fill_between(x, (y-ci), (y+ci), color='b', alpha=.2)
y = average_corr_rh
ci = 1.96*std_corr_rh/np.sqrt(j+1)
ax.plot(x, y, linewidth=2, color='r', label='Right hemisphere')
ax.fill_between(x, (y-ci), (y+ci), color='r', alpha=.2)
plt.ylabel('Correlation', fontdict={'fontsize':20})
plt.xlabel(r'Mode $\psi$', fontdict={'fontsize':20})
plt.title('Early psychosis cohort correlation across modes', fontdict={'fontsize':20})
plt.xlim([1, 200])
plt.ylim([0, 1.0])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=15)
plt.show()

# plot a few examples to show

j = 175
texture = P_Psi_lh[0,j]
view = 'lateral'
vmin = np.min(texture)
vmax = np.max(texture)
hemi = 'left'
cmap = plt.get_cmap('coolwarm')

fig = plt.figure(figsize=(15,9), constrained_layout=False)
pial_mesh = 'data/fsaverage.L.pial_orig.32k_fs_LR.surf.gii'

ax = fig.add_subplot(projection='3d')
plotting.plot_surf(
    pial_mesh, texture, hemi=hemi, 
    view=view, colorbar=False, cmap=cmap, vmin=vmin, 
    vmax=vmax, axes=ax)
ax.dist = 7

plt.show()

texture = P_Psi_lh[6,j]
view = 'lateral'
vmin = np.min(texture)
vmax = np.max(texture)
hemi = 'left'
cmap = plt.get_cmap('coolwarm')

fig = plt.figure(figsize=(15,9), constrained_layout=False)
pial_mesh = 'data/fsaverage.L.pial_orig.32k_fs_LR.surf.gii'

ax = fig.add_subplot(projection='3d')
plotting.plot_surf(
    pial_mesh, texture, hemi=hemi, 
    view=view, colorbar=False, cmap=cmap, vmin=vmin, 
    vmax=vmax, axes=ax)
ax.dist = 7

plt.show()

texture = P_Psi_lh[22,j]
view = 'lateral'
vmin = np.min(texture)
vmax = np.max(texture)
hemi = 'left'
cmap = plt.get_cmap('coolwarm')

fig = plt.figure(figsize=(15,9), constrained_layout=False)
pial_mesh = 'data/fsaverage.L.pial_orig.32k_fs_LR.surf.gii'

ax = fig.add_subplot(projection='3d')
plotting.plot_surf(
    pial_mesh, texture, hemi=hemi, 
    view=view, colorbar=False, cmap=cmap, vmin=vmin, 
    vmax=vmax, axes=ax)
ax.dist = 7

plt.show()

# =============================================================================
# #### Group differences in first 20 eigenmodes ####
# =============================================================================

HC_Psi_lh_mean = np.mean(np.abs(HC_Psi_lh), axis=0)
HC_Psi_rh_mean = np.mean(np.abs(HC_Psi_rh), axis=0)

P_Psi_lh_mean = np.mean(np.abs(P_Psi_lh), axis=0)
P_Psi_rh_mean = np.mean(np.abs(P_Psi_rh), axis=0)

n_perms=1000

zscore_array = np.zeros_like(HC_Psi_lh_mean[:20])
zscore_array_thr = np.zeros_like(zscore_array)
zscore_array_fdr_thr = np.zeros_like(zscore_array)

p = 0.05
q = 1-p/n_perms

for j in range(20):
    perm = Permutation(np.abs(HC_Psi_lh[:,j]), np.abs(P_Psi_lh[:,j]), seed=42, n_jobs=10)
    surrs = perm(n_perms)
    
    data_ref = HC_Psi_lh_mean[j] - P_Psi_lh_mean[j]
    data_all = np.mean(surrs[0], axis=1).squeeze() - np.mean(surrs[1], axis=1).squeeze()
    mean_data = np.mean(data_all, axis=0)
    std_data = np.std(data_all, axis=0)
    zscore = np.divide((data_ref - mean_data), std_data)
    
    pval_sup = np.sum([(data_ref >= i) for i in data_all], axis=0)/n_perms
    pval_inf = np.sum([(data_ref <= i) for i in data_all], axis=0)/n_perms
    
    zscore_thr = np.where(pval_inf >= p, zscore, 0.) + np.where(pval_sup >= p, zscore, 0.)
    zscore_fdr_thr = np.where(pval_inf >= q, zscore, 0.) + np.where(pval_sup >= p, zscore, 0.)
    
    zscore_array[j] = zscore
    zscore_array_thr[j] = zscore_thr
    zscore_array_fdr_thr[j] = zscore_fdr_thr
    
    
# =============================================================================
# #### Plot LBOs for each group ####
# =============================================================================

avg_method = 'mean'
cmap = plt.get_cmap('coolwarm')

# create figure
fig = plt.figure(figsize=(25,18), constrained_layout=False)
grid = gridspec.GridSpec(
    4, 5, left=0., right=1., bottom=0., top=1.2,
    height_ratios=[1,1,1,1], width_ratios=[1,1,1,1,1],
    hspace=0.0, wspace=0.0)

task = 'resting-state'
#img = image.load_img(f'{codeFolder}/masks/subcortex_mask.nii')
stri_file = f'{codeFolder}/masks/striatum_cropped.nii'
stri_msk = image.load_img(stri_file)
#sub_msk = image.load_img(f'{codeFolder}/masks/subcortex_mask.nii')
fslr = fetch_atlas('fsLR', '32k')
fontcolor = 'black'

hemi = 'left'
textures_lh = HC_Psi_lh_mean[:20] #change to fit purpose
textures_rh = HC_Psi_rh_mean[:20]
view = 'lateral'

grps = ['HC','P']
grps_label = ['Healthy Cohort (HC)', 'Early Psychosis (EP)']
view_label = ['Lateral view', 'Medial view']
grp = 'HC'

for mode in range(20):
    if hemi == 'left':
        infl_mesh = fslr.inflated[0]
        
        texture = textures_lh[mode]
        vmin, vmax = min(texture), max(texture)
        
        # for col in [1, 2]:
        #     if col == 1:
        #         view = 'lateral'
        #     else:
        #         view = 'medial'
           
        ax = fig.add_subplot(grid[mode], projection='3d')
        plotting.plot_surf(
            infl_mesh, texture, hemi=hemi, view=view,
            colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
            axes=ax)
        ax.dist = 7
        
    else:
        infl_mesh = fslr.inflated[1]
        
        texture = textures_rh[mode]
        vmin, vmax = min(texture), max(texture)
        
        # for col in [1, 2]:
        #     if col == 1:
        #         view = 'lateral'
        #     else:
        #         view = 'medial'
           
        ax = fig.add_subplot(grid[mode], projection='3d')
        plotting.plot_surf(
            infl_mesh, texture, hemi=hemi, view=view,
            colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
            axes=ax)
        ax.dist = 7
        
    label = f'Mode {mode+1}'
    ax = fig.add_subplot(grid[mode])
    ax.axis('off')
    ax.text(0.5, 0, label, ha='center', fontdict={'fontsize':30})

if grp == 'HC':
    ax = fig.add_subplot(grid[1])
    ax.axis('off')    
    ax.text(1.5, 1, f'{grps_label[0]}', ha='center', 
            fontdict={'fontsize':30, 'color':fontcolor})

else:
    ax = fig.add_subplot(grid[1])
    ax.axis('off')    
    ax.text(1.5, 1, f'{grps_label[1]}', ha='center', 
            fontdict={'fontsize':30, 'color':fontcolor})
 
# colorbar
cax = plt.axes([1.05, 0.32, 0.03, 0.4])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.set_ticks([])
cbar.ax.set_title('max', fontdict={'fontsize':30}, pad=20)
cbar.ax.set_xlabel('min', fontdict={'fontsize':30}, labelpad=20)

plt.show()
    
# =============================================================================
# #### Plot differences ####
# =============================================================================

avg_method = zscore_avg_method

cmap=plt.get_cmap('twilight_shifted')
twil = cm.get_cmap('twilight_shifted', 1000)
newcolors = np.vstack((twil(np.linspace(0, 0.3, 400)),
                        twil(np.linspace(0.4, 0.6, 400)),
                        twil(np.linspace(0.7, 1, 400))))
cmap = ListedColormap(newcolors, name='twilight_shifted_threshold')

fig = plt.figure(figsize=(25,18), constrained_layout=False)
grid = gridspec.GridSpec(
    4, 5, left=0., right=1., bottom=0., top=1.2,
    height_ratios=[1,1,1,1], width_ratios=[1,1,1,1,1],
    hspace=0.0, wspace=0.0)

i = 0

grps_label = ['HC', 'P']
view = 'lateral'
fontcolor='black'
hemi = 'left'

vmin, vmax = -6, 6

for mode in range(zscore_array.shape[0]):
    if hemi == 'left':
        infl_mesh = fslr.inflated[0]
        
        texture = zscore_array[mode]
        
        # for col in [1, 2]:
        #     if col == 1:
        #         view = 'lateral'
        #     else:
        #         view = 'medial'
           
        ax = fig.add_subplot(grid[mode], projection='3d')
        plotting.plot_surf(
            infl_mesh, texture, hemi=hemi, view=view,
            colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
            axes=ax)
        ax.dist = 7
        
    else:
        infl_mesh = fslr.inflated[1]
        
        texture = zscore_array[mode]
        
        # for col in [1, 2]:
        #     if col == 1:
        #         view = 'lateral'
        #     else:
        #         view = 'medial'
           
        ax = fig.add_subplot(grid[mode], projection='3d')
        plotting.plot_surf(
            infl_mesh, texture, hemi=hemi, view=view,
            colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
            axes=ax)
        ax.dist = 7
        
    i += 1
    label = f'Mode {mode+1}'
    ax = fig.add_subplot(grid[mode])
    ax.axis('off')
    ax.text(0.5, 0, label, ha='center', fontdict={'fontsize':30})
        
ax = fig.add_subplot(grid[1])
ax.axis('off')    
ax.text(1.5, 1, 'Healthy Controls (HC) - Early Psychosis (EP)', ha='center', 
        fontdict={'fontsize':30, 'color':fontcolor})
        
    
# colorbar
cax = plt.axes([1.07, 0.3, 0.05, 0.3])
cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax), cmap=cmap), cax=cax)
cbar.set_ticks([-2,2])
cbar.set_ticklabels([r'-2$\sigma$',r'2$\sigma$'])
cbar.ax.tick_params(labelsize=30, labelcolor=fontcolor)
cbar.ax.set_title(f'{grps_label[0]}>{grps_label[1]}', fontdict={'fontsize':30, 'color':fontcolor}, pad=20)
cbar.ax.set_xlabel(f'{grps_label[1]}>{grps_label[0]}', fontdict={'fontsize':30, 'color':fontcolor}, labelpad=20)

plt.show()

# =============================================================================
# #### Plot differences thresholded ####
# =============================================================================

avg_method = zscore_avg_method

cmap=plt.get_cmap('twilight_shifted')
twil = cm.get_cmap('twilight_shifted', 1000)
newcolors = np.vstack((twil(np.linspace(0, 0.3, 400)),
                        twil(np.linspace(0.4, 0.6, 400)),
                        twil(np.linspace(0.7, 1, 400))))
cmap = ListedColormap(newcolors, name='twilight_shifted_threshold')

fig = plt.figure(figsize=(25,18), constrained_layout=False)
grid = gridspec.GridSpec(
    4, 5, left=0., right=1., bottom=0., top=1.2,
    height_ratios=[1,1,1,1], width_ratios=[1,1,1,1,1],
    hspace=0.0, wspace=0.0)

i = 0

grps_label = ['HC', 'P']
view = 'lateral'
fontcolor='black'
hemi = 'left'

vmin, vmax = -6, 6

for mode in range(zscore_array.shape[0]):
    if hemi == 'left':
        infl_mesh = fslr.inflated[0]
        
        texture = zscore_array_thr[mode]
        
        # for col in [1, 2]:
        #     if col == 1:
        #         view = 'lateral'
        #     else:
        #         view = 'medial'
           
        ax = fig.add_subplot(grid[mode], projection='3d')
        plotting.plot_surf(
            infl_mesh, texture, hemi=hemi, view=view,
            colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
            axes=ax)
        ax.dist = 7
        
    else:
        infl_mesh = fslr.inflated[1]
        
        texture = zscore_array_thr[mode]
        
        # for col in [1, 2]:
        #     if col == 1:
        #         view = 'lateral'
        #     else:
        #         view = 'medial'
           
        ax = fig.add_subplot(grid[mode], projection='3d')
        plotting.plot_surf(
            infl_mesh, texture, hemi=hemi, view=view,
            colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
            axes=ax)
        ax.dist = 7
        
    i += 1
    label = f'Mode {mode+1}'
    ax = fig.add_subplot(grid[mode])
    ax.axis('off')
    ax.text(0.5, 0, label, ha='center', fontdict={'fontsize':30})
        
ax = fig.add_subplot(grid[1])
ax.axis('off')    
ax.text(1.5, 1, 'Healthy Controls (HC) - Early Psychosis (EP) threshold at p<0.05', ha='center', 
        fontdict={'fontsize':30, 'color':fontcolor})
        
    
# colorbar
cax = plt.axes([1.07, 0.3, 0.05, 0.3])
cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax), cmap=cmap), cax=cax)
cbar.set_ticks([-2,2])
cbar.set_ticklabels([r'-2$\sigma$',r'2$\sigma$'])
cbar.ax.tick_params(labelsize=30, labelcolor=fontcolor)
cbar.ax.set_title(f'{grps_label[0]}>{grps_label[1]}', fontdict={'fontsize':30, 'color':fontcolor}, pad=20)
cbar.ax.set_xlabel(f'{grps_label[1]}>{grps_label[0]}', fontdict={'fontsize':30, 'color':fontcolor}, labelpad=20)


plt.show()

# =============================================================================
# #### Plot differences FDR thresholded ####
# =============================================================================

avg_method = zscore_avg_method

cmap=plt.get_cmap('twilight_shifted')
twil = cm.get_cmap('twilight_shifted', 1000)
newcolors = np.vstack((twil(np.linspace(0, 0.3, 400)),
                        twil(np.linspace(0.4, 0.6, 400)),
                        twil(np.linspace(0.7, 1, 400))))
cmap = ListedColormap(newcolors, name='twilight_shifted_threshold')

fig = plt.figure(figsize=(25,18), constrained_layout=False)
grid = gridspec.GridSpec(
    4, 5, left=0., right=1., bottom=0., top=1.2,
    height_ratios=[1,1,1,1], width_ratios=[1,1,1,1,1],
    hspace=0.0, wspace=0.0)

i = 0

grps_label = ['HC', 'P']
view = 'lateral'
fontcolor='black'
hemi = 'left'

vmin, vmax = -6, 6

for mode in range(zscore_array.shape[0]):
    if hemi == 'left':
        infl_mesh = fslr.inflated[0]
        
        texture = zscore_array_fdr_thr[mode]
        
        # for col in [1, 2]:
        #     if col == 1:
        #         view = 'lateral'
        #     else:
        #         view = 'medial'
           
        ax = fig.add_subplot(grid[mode], projection='3d')
        plotting.plot_surf(
            infl_mesh, texture, hemi=hemi, view=view,
            colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
            axes=ax)
        ax.dist = 7
        
    else:
        infl_mesh = fslr.inflated[1]
        
        texture = zscore_array_fdr_thr[mode]
        
        # for col in [1, 2]:
        #     if col == 1:
        #         view = 'lateral'
        #     else:
        #         view = 'medial'
           
        ax = fig.add_subplot(grid[mode], projection='3d')
        plotting.plot_surf(
            infl_mesh, texture, hemi=hemi, view=view,
            colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
            axes=ax)
        ax.dist = 7
        
    i += 1
    label = f'Mode {mode+1}'
    ax = fig.add_subplot(grid[mode])
    ax.axis('off')
    ax.text(0.5, 0, label, ha='center', fontdict={'fontsize':30})
        
ax = fig.add_subplot(grid[1])
ax.axis('off')    
ax.text(1.5, 1, r'Healthy Controls (HC) - Early Psychosis (EP) threshold at $p_{FDR}$<0.05', ha='center', 
        fontdict={'fontsize':30, 'color':fontcolor})
        
    
# colorbar
cax = plt.axes([1.07, 0.3, 0.05, 0.3])
cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax), cmap=cmap), cax=cax)
cbar.set_ticks([-2,2])
cbar.set_ticklabels([r'-2$\sigma$',r'2$\sigma$'])
cbar.ax.tick_params(labelsize=30, labelcolor=fontcolor)
cbar.ax.set_title(f'{grps_label[0]}>{grps_label[1]}', fontdict={'fontsize':30, 'color':fontcolor}, pad=20)
cbar.ax.set_xlabel(f'{grps_label[1]}>{grps_label[0]}', fontdict={'fontsize':30, 'color':fontcolor}, labelpad=20)


plt.show()


# =============================================================================
# #### Betas ####
# =============================================================================

grp_folder = 'cohorts'
grps = ['HC','EP']
grps_label = ['Healthy Cohort (HC)', 'Early Psychosis (EP)']

fig = plt.figure(figsize=(25, 15), constrained_layout=False)
grid = gridspec.GridSpec(
    1, 2, left=0., right=1., bottom=0., top=1.,
    height_ratios=[1.], width_ratios=[1.,1.],
    hspace=0.0, wspace=0.05)

Vn = 2
 
i = 0
# plot betas per group
for grp in grps:
    betas_subwise_lh = np.load(f'data/HCP-EP_{grp}_Vn{Vn}_stdized_betas_lh.npy')
    betas_subwise_rh = np.load(f'data/HCP-EP_{grp}_Vn{Vn}_stdized_betas_rh.npy')
    
    betas_lh = np.nanmean(betas_subwise_lh, axis=0)
    betas_rh = np.nanmean(betas_subwise_rh, axis=0)
    
    ax = fig.add_subplot(grid[i])
    markerline, _, _ = plt.stem(np.arange(1,len(betas_lh)+1), betas_lh, linefmt='blue', markerfmt='D', label='LH')
    markerline.set_markerfacecolor('none')
    markerline, _, _ = plt.stem(np.arange(1,len(betas_rh)+1), betas_rh, linefmt='red', markerfmt='D', label='RH')
    markerline.set_markerfacecolor('none')
    #markerline, _, _ = plt.stem(np.arange(1,len(zscore)+1), zscore_thr_plot, linefmt='black', markerfmt='D', label='Z-score (p<0.05)')
    #markerline.set_markerfacecolor('none')
    plt.xlabel(r'Geometric mode $\phi_j$', fontdict={'fontsize':20})
    if i != 1:
        plt.ylabel(r'Coefficient $\beta_j$', fontdict={'fontsize':20})
    plt.title(fr'{grps_label[i]} - Average $\beta_j$ estimates of $\phi_{Vn-1}$', fontdict={'fontsize':20})
    plt.legend(fontsize='20')
    plt.ylim((-1000,1000))
    plt.xlim((0,200))
    i += 1

plt.show()

# =============================================================================
# #### Betas with null testing ####
# =============================================================================

burt_nulls = np.load(f'data/HCP-EP_{grp}_V{Vn}_burt_nulls.npy')

burt_nulls_lh = burt_nulls[:,:burt_nulls.shape[2]//2]
burt_nulls_rh = burt_nulls[:,burt_nulls.shape[2]//2:]

# run model for each null to produce a spectra of null betas

Psi_lh = HC_Psi_lh
Psi_rh = HC_Psi_rh

betas_nulls_lh = np.zeros((burt_nulls.shape[1], betas_subwise_lh.shape[0], betas_subwise_lh.shape[1]))
betas_nulls_rh = np.zeros((burt_nulls.shape[1], betas_subwise_rh.shape[0], betas_subwise_rh.shape[1]))

for p in range(burt_nulls_lh.shape[0]):
    # optimize eigenmode order
    Psi_opt_lh = np.vstack(maximise_recon_metric(Psi_lh[sub], burt_nulls_lh[p])
                        for sub in range(Psi_lh.shape[0])).reshape(Psi_lh.shape)

    Psi_opt_rh = np.vstack(maximise_recon_metric(Psi_rh[sub], burt_nulls_lh[p])
                        for sub in range(Psi_rh.shape[0])).reshape(Psi_rh.shape)
    
    betas_nulls_lh[p] = np.vstack(np.matmul(Psi_opt_lh[sub], burt_nulls_lh[p]) for sub in range(Psi_lh.shape[0]))
    betas_nulls_rh[p] = np.vstack(np.matmul(Psi_opt_rh[sub], burt_nulls_rh[p]) for sub in range(Psi_rh.shape[0]))

betas_nulls_lh_mean = betas_nulls_lh.mean(axis=0)
beta_nulls_lh = betas_nulls_lh_mean.mean(axis=0)

betas_nulls_rh_mean = betas_nulls_rh.mean(axis=0)
beta_nulls_rh = betas_nulls_rh_mean.mean(axis=0)

zscore_lh = (betas_subwise_lh.mean(axis=0) - beta_nulls_lh.mean())/beta_nulls_lh.std()
    
pval_sup_lh = np.sum([(betas_subwise_lh.mean(axis=0) >= i) for i in beta_nulls_lh], axis=0)/n_perms
pval_inf_lh = np.sum([(betas_subwise_lh.mean(axis=0) >= i) for i in beta_nulls_lh], axis=0)/n_perms
    
zscore_lh_thr = np.where(pval_inf_lh >= 0.95, zscore_lh, 0.) + np.where(pval_sup_lh >= 0.95, zscore_lh, 0.)

zscore_rh = (betas_subwise_rh.mean(axis=0) - beta_nulls_rh.mean())/beta_nulls_rh.std()
    
pval_sup_rh = np.sum([(betas_subwise_rh.mean(axis=0) >= i) for i in beta_nulls_rh], axis=0)/n_perms
pval_inf_rh = np.sum([(betas_subwise_rh.mean(axis=0) >= i) for i in beta_nulls_rh], axis=0)/n_perms
    
zscore_rh_thr = np.where(pval_inf >= 0.95, zscore_rh, 0.) + np.where(pval_sup >= 0.95, zscore_rh, 0.)


# plot
grp_folder = 'cohorts'
grps = ['HC','EP']
grps_label = ['Healthy Cohort (HC)', 'Early Psychosis (EP)']

fig = plt.figure(figsize=(25, 15), constrained_layout=False)
grid = gridspec.GridSpec(
    1, 2, left=0., right=1., bottom=0., top=1.,
    height_ratios=[1.], width_ratios=[1.,1.],
    hspace=0.0, wspace=0.05)

Vn = 2
 
i = 0
# plot betas per group

ax = fig.add_subplot()
markerline, _, _ = plt.stem(np.arange(1,len(zscore_lh)+1), zscore_lh, linefmt='blue', markerfmt='D', label='LH')
markerline.set_markerfacecolor('none')
markerline, _, _ = plt.stem(np.arange(1,len(zscore_rh)+1), zscore_rh, linefmt='red', markerfmt='D', label='RH')
markerline.set_markerfacecolor('none')
#markerline, _, _ = plt.stem(np.arange(1,len(zscore)+1), zscore_thr_plot, linefmt='black', markerfmt='D', label='Z-score (p<0.05)')
#markerline.set_markerfacecolor('none')
plt.xlabel(r'Geometric mode $\phi_j$', fontdict={'fontsize':20})
if i != 1:
    plt.ylabel(r'Coefficient $\beta_j$', fontdict={'fontsize':20})
plt.title(fr'{grps_label[i]} - Average $\beta_j$ estimates of $\phi_{Vn-1}$', fontdict={'fontsize':20})
plt.legend(fontsize='20')
plt.ylim((-10,10))
plt.xlim((0,200))

plt.show()

# =============================================================================
# #### Group difference maps in the model ####
# =============================================================================


