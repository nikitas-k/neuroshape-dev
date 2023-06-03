#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nilearn import plotting, image, masking  
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import surface
from matplotlib import gridspec
import numpy as np
import pyvista as pv
from tqdm import tqdm
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize
from neuromaps import transforms


##############################
# HIPPOCAMPUS/SUBCORTEX MESH #
##############################

codeFolder = '/Volumes/Scratch/functional_integration_psychosis/code/subcortex/'
# create point cloud
# cd /home/leonie/Documents/source/gradientography-task-fmri
#mask_file = 'masks/hippocampus_cropped.nii'
mask_file = f'{codeFolder}/masks/striatum_cropped.nii'
# mask_file = 'masks/subcortex_mask_part1_cropped.nii'

msk = image.load_img(mask_file)
msk_data = msk.get_fdata()
affine = msk.affine

xlist, ylist, zlist = [], [], []
for x in tqdm(range(msk_data.shape[0])):
    for y in range(msk_data.shape[1]):
        for z in range(msk_data.shape[2]):
            if msk_data[x,y,z] == 1:
                select = False
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        for k in [-1, 0, 1]:
                            if x+i < msk_data.shape[0] and x+i >= 0 and \
                               y+j < msk_data.shape[1] and y+j >= 0 and \
                               z+k < msk_data.shape[2] and z+k >= 0:
                                if msk_data[x+i,y+j,z+k] != 1:
                                    select = True
                            else:
                                select = True
                if select:
                    (xa, ya, za) = image.coord_transform(x, y, z, affine)
                    xlist.append(xa)
                    ylist.append(ya)
                    zlist.append(za)
points = np.array([xlist, ylist, zlist]).T

# convert point cloud in surface
cloud = pv.PolyData(points)
volume = cloud.delaunay_3d(alpha=3)
shell = volume.extract_geometry()
smooth = shell.smooth(n_iter=100, relaxation_factor=0.01,
                      feature_smoothing=False, 
                      boundary_smoothing=True,
                      edge_angle=100, feature_angle=100)

# extract faces
faces = []
i, offset = 0, 0
cc = smooth.faces 
while offset < len(cc):
    nn = cc[offset]
    faces.append(cc[offset+1:offset+1+nn])
    offset += nn + 1
    i += 1

# convert to triangles
triangles = []
for face in faces:
    if len(face) == 3:
        triangles.append(face)
    elif len(face) == 4:
        triangles.append(face[:3])
        triangles.append(face[-3:])
    else:
        print(len(face))

# create mesh
mesh = [smooth.points, np.array(triangles)]

##############################
# FIGURE. GRADIENTS EIGENMAP #
##############################

resultFolder = '/Volumes/Scratch/functional_integration_psychosis/code/subcortex/result/microres'
# resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_nimg'

cmap = plt.get_cmap('viridis')
grp_folder = 'cohorts'
grps = ['HC','EP']
grps_label = ['Healthy Cohort (HC)', 'Early Psychosis (EP)']

fig = plt.figure(figsize=(21,25), constrained_layout=False)
grid = gridspec.GridSpec(
    5, 5, left=0., right=1., bottom=0., top=1.,
    height_ratios=[0.4,1,1,1,1], width_ratios=[0.2,1,1,1,1],
    hspace=0.0, wspace=0.0)

i = 6
for task in ['go', 'nogo', 'background']:
    for grp in grps:
        for Vn in [2, 3]:
            # texture
            eig_file = f'{resultFolder}/tasks/{task}/{grp_folder}/{grp}/Vn{Vn}_eigenvector.nii'
            eig = image.load_img(eig_file)
            texture = surface.vol_to_surf(
                eig, mesh, interpolation='nearest', radius=3)
            
            # plot
            ax = fig.add_subplot(grid[i], projection='3d')
            print(f'{task} {grp} {Vn} : {max(texture):.3f}')
            plotting.plot_surf(mesh, texture, view='anterior', vmin=0, vmax=max(texture),
                                cmap=cmap, avg_method='mean',
                                axes=ax)
            ax.text(-25, -30, 17, 'L', va='center', fontdict={'fontsize':30})
            ax.text(25, -30, 17, 'R', va='center', fontdict={'fontsize':30})
            
            # variance explained
            var = pd.read_csv(f'{resultFolder}/tasks/{task}/{grp_folder}/{grp}/variance_explained.csv',
                              header=None)
            ax.text(45, -15, 0, f'{var.loc[Vn-2,0]:.2f}%', va='center', fontdict={'fontsize':30})
            i += 1
    i += 1

# add text
for label, id_grid in zip(['GO', 'NOGO', 'BACKGROUND'], [5, 10, 15]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')    
    ax.text(0.3, 0.5, label, rotation=90, 
            va='center', fontdict={'fontsize':30})

for label, id_grid in zip(grps_label, [1, 3]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')
    ax.text(1, 0.5, label, ha='center', fontdict={'fontsize':30})
    ax.text(0.5, 0, 'Gradient I', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid+1])
    ax.axis('off')
    ax.text(0.5, 0, 'Gradient II', ha='center', fontdict={'fontsize':30})

# colorbar
cax = plt.axes([1.01, 0.3, 0.03, 0.3])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.set_ticks([])
cbar.ax.set_title('max', fontdict={'fontsize':30}, pad=20)
cbar.ax.set_xlabel('0', fontdict={'fontsize':30}, labelpad=20)

plt.show()

################################
# FIGURE. GRADIENT I MAGNITUDE #
################################

resultFolder = 'result/microres'
# resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_nimg'
cmap = plt.get_cmap('viridis')
grp_folder = 'cohorts'
grps = ['HC', 'EP']
grps_label = ['Healthy Cohort (HC)', 'Early Psychosis (EP)']

fontcolor = 'black'

fig = plt.figure(figsize=(21,16), constrained_layout=False, dpi=300)
grid = gridspec.GridSpec(
    4, 5, left=0., right=1., bottom=0., top=1.,
    height_ratios=[0.2,1,1,1], width_ratios=[0.2,1,1,1,1],
    hspace=0.0, wspace=0.0)

i = 6
Vn = 2
for task in ['go', 'nogo', 'background']:
    for grp in grps:
        # for Vn in [2, 3]:
        # texture
        mag_file = f'{resultFolder}/tasks/{task}/{grp_folder}/{grp}/Vn{Vn}_magnitude.nii'
        # mag_file = f'{resultFolder}/tasks/{task}/cohort/Vn{Vn}_{grp}_mean_boot.nii'
        mag = image.load_img(mag_file)
        texture = surface.vol_to_surf(
            mag, mesh, interpolation='nearest', radius=3, mask_img=mask_file)
        for view in ['dorsal', 'ventral']:
            # plot
            ax = fig.add_subplot(grid[i], projection='3d')
            plotting.plot_surf(mesh, texture, view=view, vmin=0, vmax=0.5,
                                cmap=cmap, avg_method='mean',
                                axes=ax)
            if view == 'dorsal':
                ax.text(37, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':fontcolor})
                ax.text(-35, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':fontcolor})
                if task == 'resting_state':
                    ax.text(40, -15, 53, 'dorsal view', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            else:
                ax.text(38, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':'white'})
                ax.text(-36, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':'white'})
                if task == 'resting_state':
                    ax.text(0, 0, 0, 'ventral view', va='center', fontdict={'fontsize':30, 'color':'white'})
                ax.set_facecolor('lightslategrey') #(0.4,0.4,0.4))
            i += 1
    i += 1

# add text
for label, id_grid in zip(['GO', 'NOGO', 'BACKGROUND'], [5, 10, 15]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')    
    ax.text(0.3, 0.5, label, rotation=90, 
            va='center', fontdict={'fontsize':30})

for label, id_grid in zip(grps_label, [1, 3]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')
    ax.text(1, 0.4, label, ha='center', fontdict={'fontsize':30, 'color':fontcolor})
    ax = fig.add_subplot(grid[id_grid+1])
    ax.axis('off')

# colorbar
import matplotlib.cm as cm
cax = plt.axes([1.01, 0.3, 0.03, 0.3])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.set_ticks([])
cbar.ax.set_title('0.2', fontdict={'fontsize':30, 'color':fontcolor}, pad=20)
cbar.ax.set_xlabel('0', fontdict={'fontsize':30, 'color':fontcolor}, labelpad=20)

plt.show()
fig.savefig(f'/tmp/magnitude_Vn{Vn}.png', dpi=300)

#############################
# FIGURE. COHORT COMPARISON #
#############################

#resultFolder = 'result'
# resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_nimg'

# projection parameters
interpolation='nearest'
kind='line'
radius=3
def custom_function(vertices):
    val = 0.0
    for v in vertices:
        if np.abs(v) > np.abs(val):
            val = v
    return val
avg_method = custom_function # define function that take the max(|v|)
#avg_method = 'mean'

grp_folder = 'cohorts'
grps = ['HC', 'EP']
grps_label = ['HC', 'EP']

fontcolor = 'darkslategrey'
fontcolor='black'

cmap=plt.get_cmap('twilight_shifted')
twil = cm.get_cmap('twilight_shifted', 1000)
newcolors = np.vstack((twil(np.linspace(0, 0.3, 400)),
                        twil(np.linspace(0.4, 0.6, 400)),
                        twil(np.linspace(0.7, 1, 400))))
cmap = ListedColormap(newcolors, name='twilight_shifted_threshold')

fig = plt.figure(figsize=(11,14), constrained_layout=False, dpi=300)
grid = gridspec.GridSpec(
    4, 3, left=0., right=1., bottom=0., top=1.,
    height_ratios=[0.2,1,1,1], width_ratios=[0.2,1,1],
    hspace=0.0, wspace=0.0)

i = 4
Vn = 2
for task in ['go','nogo','background']: #, 'nogo', 'resting-state']:
    # texture
    mag_file = f'{resultFolder}/tasks/{task}/{grp_folder}/Vn{Vn}_z_{grps[1]}-{grps[0]}.nii'
    mag = image.load_img(mag_file)
    #mag_msk = f'{resultFolder}/tasks/{task}/{grp_folder}/Vn{Vn}_pval.nii'
    #mag_1d = masking.apply_mask(mag, mag_msk)
    #mag_thr = masking.unmask(mag_1d, mag_msk)
    
    texture = surface.vol_to_surf(
        mag, mesh, interpolation=interpolation, radius=radius, 
        kind=kind, mask_img=mask_file)
    
    vmax = 6.0
    for view in ['dorsal', 'ventral']:
        # plot
        ax = fig.add_subplot(grid[i], projection='3d')
        plotting.plot_surf(mesh, texture, view=view, vmin=-vmax, vmax=vmax,
                            cmap=cmap, avg_method=avg_method,
                            axes=ax)
        if view == 'dorsal':
            ax.text(37, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            ax.text(-35, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            #ax.text(40, -15, 53, 'dorsal view', va='center', fontdict={'fontsize':30, 'color':fontcolor})     
        else:
            ax.text(38, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':'white'})
            ax.text(-38, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':'white'})
            ax.set_facecolor('lightslategrey')
            #ax.text(-40, -22, -38, 'ventral view', va='center', fontdict={'fontsize':30, 'color':'white'})
        i += 1
    i += 1

# add text
for label, id_grid in zip(['GO', 'NOGO', 'BACKGROUND'], [3, 6, 9]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')    
    ax.text(0.3, 0.5, label, rotation=90, 
            va='center', fontdict={'fontsize':30})

ax = fig.add_subplot(grid[1])
ax.axis('off')    
ax.text(1, 0.4, 'Healthy Controls (HC) - Early Psychosis (EP)', ha='center', 
        fontdict={'fontsize':30, 'color':fontcolor})

# colorbar
cax = plt.axes([1.07, 0.3, 0.05, 0.3])
cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax), cmap=cmap), cax=cax)
cbar.set_ticks([-2,2])
cbar.set_ticklabels([r'-2$\sigma$',r'2$\sigma$'])
cbar.ax.tick_params(labelsize=30, labelcolor=fontcolor)
cbar.ax.set_title(f'{grps_label[1]}>{grps_label[0]}', fontdict={'fontsize':30, 'color':fontcolor}, pad=20)
cbar.ax.set_xlabel(f'{grps_label[0]}>{grps_label[1]}', fontdict={'fontsize':30, 'color':fontcolor}, labelpad=20)

plt.show()
fig.savefig(f'/tmp/cohort_comparison_Vn{Vn}.png', dpi=300)


###########################
# FIGURE. TASK COMPARISON #
###########################

resultFolder = 'result'
# resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_TSTAT_with_confounds'

# projection parameters
interpolation='nearest'
kind='line'
radius=3
def custom_function(vertices):
    val = 0
    for v in vertices:
        if abs(v) > abs(val):
            val = v
    return val
avg_method = custom_function # define function that take the max(|v|)

from matplotlib import cm
from matplotlib.colors import ListedColormap
twil = cm.get_cmap('twilight_shifted', 1000)
newcolors = np.vstack((twil(np.linspace(0, 0.3, 400)),
                        twil(np.linspace(0.4, 0.6, 400)),
                        twil(np.linspace(0.7, 1, 400))))
cmap_comp = ListedColormap(newcolors, name='twilight_shifted_threshold')

fig = plt.figure(figsize=(21,6), constrained_layout=False, dpi=300)
grid = gridspec.GridSpec(
    2, 5, left=0., right=1., bottom=0., top=1.,
    height_ratios=[0.2,1], width_ratios=[0.2,1,1,1,1],
    hspace=0.0, wspace=0.0)

i = 6
for grp in ['hc', 'cc']:
    # texture
    mag_file = f'{resultFolder}/cohort/{grp}/tasks/Vn{Vn}_z_continuing-naive.nii'
    mag = image.load_img(mag_file)
    texture = surface.vol_to_surf(
        mag, mesh, interpolation=interpolation, radius=radius,
        kind=kind, mask_img=mask_file)
    vmax = 6
    for view in ['anterior', 'posterior']:
        # plot
        ax = fig.add_subplot(grid[i], projection='3d')
        plotting.plot_surf(mesh, texture, view=view, vmin=-vmax, vmax=vmax,
                            cmap=cmap_comp, avg_method=avg_method,
                            axes=ax)
        if view == 'anterior':
            ax.text(32, 0, 15, 'L', va='center', fontdict={'fontsize':30})
            ax.text(-25, 0, 15, 'R', va='center', fontdict={'fontsize':30})
            ax.text(25, 0, -30, 'anterior view', va='center', fontdict={'fontsize':30})
        else:
            ax.text(25, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':'white'})
            ax.text(-32, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':'white'})
            ax.set_facecolor('lightslategrey')
            ax.text(-25, 0, -30, 'posterior view', va='center', fontdict={'fontsize':30, 'color':'white'})
        i += 1

# add text
ax = fig.add_subplot(grid[5])
ax.axis('off')
ax.text(0.3, 0.5, 'CONT - NAIVE', rotation=90, 
        va='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[1])
ax.axis('off')
ax.text(1, 0.4, 'Healthy Cohort', ha='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[2])
ax.axis('off')
ax = fig.add_subplot(grid[3])
ax.axis('off')
ax.text(1, 0.4, 'Clinical Cohort', ha='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[4])
ax.axis('off')

# colorbar
cax = plt.axes([1.08, 0.2, 0.03, 0.4])
cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax), cmap=cmap_comp), cax=cax)
cbar.set_ticks([-2,2])
cbar.set_ticklabels([r'-2$\sigma$',r'2$\sigma$'])
cbar.ax.tick_params(labelsize=30)
cbar.ax.set_title('CONT>NAIVE', fontdict={'fontsize':30}, pad=20)
cbar.ax.set_xlabel('NAIVE>CONT', fontdict={'fontsize':30}, labelpad=20)

plt.show()
fig.savefig(f'/tmp/task_comparison_Vn{Vn}.png', bbox_inches='tight', dpi=300)

###############################################
# FIGURE. CORTICAL PROJECTION - HC CONTINUING #
###############################################

# =============================================================================
# #### GET FUNCTIONAL GRADIENTS HC ####
# =============================================================================

codeFolder = '/Volumes/Scratch/functional_integration_psychosis/code/subcortex'
resultFolder = f'{codeFolder}/result/microres'

grp = 'HC'
grp_folder = 'cohorts'
task = 'background'
Vn = 2

fontcolor='black'
    
stri_file = f'{codeFolder}/masks/striatum_2mm.nii'
stri_msk = image.load_img(stri_file)
stri_msk_cropped = image.load_img(f'{codeFolder}/masks/striatum_cropped.nii')
stri_side = 'left'
    
eig_file = f'{resultFolder}/tasks/{task}/cohorts/{grp}/Vn{Vn}_eigenvector.nii'
# img_file = f'{resultFolder}/projection/boot/{task}/1_eigenvector_projection_{hip[:-4]}_fix.nii'
# eig_file = f'{resultFolder}/projection/boot/{task}/1_Vn{Vn}_eigenvector.nii'
eig = image.load_img(eig_file)

eig_1d = masking.apply_mask(eig, stri_msk_cropped)
eig_mni = masking.unmask(eig_1d, stri_msk)
eig_stri_1d = masking.apply_mask(eig_mni, stri_msk)
eig_stri = masking.unmask(eig_stri_1d, stri_msk)

vmin = min(eig_stri_1d)
vmax = max(eig_stri_1d)

neuroshapeFolder = '/Volumes/Scratch/functional_integration_psychosis/code/neuroshape'

interpolation='linear'
kind='ball'
radius=3
n_samples = None
mask_img = f'{codeFolder}/masks/cortex.nii'
row = 0
cmap = cmap

text_file = f'{resultFolder}/tasks/{task}/cohorts/{grp}/subjects.txt'
subjects = np.loadtxt(text_file, dtype='str').squeeze()

HC_Phi_lh = np.zeros((len(subjects), 32492))
HC_Phi_rh = np.zeros((len(subjects), 32492))

i = 0
for subject in subjects:
    img_file = f'{resultFolder}/tasks/{task}/{grp_folder}/{grp}/{subject}_Vn{Vn}_eigenvector_projection_striatum_2mm.nii'
    img = image.load_img(img_file)
    
    func_surf_fslr = transforms.mni152_to_fslr(img, '32k')
    func_surf_lh = func_surf_fslr[0].agg_data()
    func_surf_rh = func_surf_fslr[1].agg_data()
    
    HC_Phi_lh[i] = func_surf_lh
    HC_Phi_rh[i] = func_surf_rh
        
    i += 1
        
texture_lh = HC_Phi_lh
texture_rh = HC_Phi_rh

# create figure
fig = plt.figure(figsize=(30,9), constrained_layout=False)
grid = gridspec.GridSpec(
    2, 5, left=0., right=1., bottom=0., top=1.,
    height_ratios=[1.,1.], width_ratios=[0.6,1.2,1.2,1.2,1.2],
    hspace=0.0, wspace=0.0)

view='lateral'

print(f'vmin={vmin}, vmax={vmax}')

texture_stri = surface.vol_to_surf(
    eig, mesh, interpolation='linear', radius=5, mask_img=mask_file)

for sub in [0, 1, 2, 3]:
    for hemi in ['left', 'right']:
        if hemi == 'left':
            pial_mesh = f'{neuroshapeFolder}/data/fsaverage.L.pial_orig.32k_fs_LR.surf.gii'
            
            texture = texture_lh[sub+3]
            
            ax = fig.add_subplot(grid[sub+1], projection='3d')
            plotting.plot_surf(
                pial_mesh, texture, hemi=hemi, view=view,
                colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
                axes=ax, avg_method='mean')
            ax.dist = 7
        
            # plot striatum
            # ax = fig.add_subplot(grid[0])
            # disp = plotting.plot_img(
            #     eig_stri, display_mode='z', threshold=0, cmap=cmap, 
            #     vmin=vmin_stri, vmax=vmax_stri,
            #     axes=ax, cut_coords=cut_coords, colorbar=False, annotate=False)
            # disp.annotate(size=25)
            # texture_stri = surface.vol_to_surf(
            #     eig, mesh, interpolation='nearest', radius=3, mask_img=mask_file)
        
            # plot
            ax = fig.add_subplot(grid[0], projection='3d')
            plotting.plot_surf(mesh, texture_stri, view='dorsal', vmin=0, vmax=max(texture_stri),
                                cmap=cmap, avg_method='mean',
                                axes=ax)
            
            ax.text(-20,-35, 15, 'L', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            ax.text(20, -35, 15, 'R', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            #ax.text(100, 0, 0, text, ha='center', fontdict={'fontsize':30})
            ax.text(40, -18, 0, 'dorsal view', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            
        else:
            pial_mesh = f'{neuroshapeFolder}/data/fsaverage.R.pial_orig.32k_fs_LR.surf.gii'
            
            texture = texture_rh[sub]
            
            ax = fig.add_subplot(grid[sub+6], projection='3d')
            plotting.plot_surf(
                pial_mesh, texture, hemi=hemi, view=view,
                colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
                axes=ax, avg_method='mean')
            ax.dist = 7
            
            # plot striatum
            # ax = fig.add_subplot(grid[3])
            # disp = plotting.plot_img(
            #     eig_stri, display_mode='y', threshold=0, cmap=cmap, 
            #     vmin=vmin, vmax=vmax,
            #     axes=ax, cut_coords=cut_coords, colorbar=False, annotate=False)
            # disp.annotate(size=25)
            #texture_stri = surface.vol_to_surf(
                #eig, mesh, interpolation='nearest', radius=3, mask_img=mask_file)
        
            # plot
            ax = fig.add_subplot(grid[5], projection='3d')
            plotting.plot_surf(mesh, texture_stri, view='ventral', vmin=0, vmax=max(texture_stri),
                                cmap=cmap, avg_method='mean',
                                axes=ax)
            ax.text(-20, -35, 15, 'L', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            ax.text(20, -35, 15, 'R', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            #ax.text(-40, 0, 0, text, ha='center', fontdict={'fontsize':30})
            ax.text(-40, -18, 0, 'ventral view', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            #ax.set_facecolor('lightslategrey') #(0.4,0.4,0.4))
        
    
for label, id_grid in zip([f'Gradient {Vn-1} : {task} - {grp}'], [2]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')
    ax.text(0.8, 1, label, ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid+4])
    ax.axis('off')
    ax.text(0.5, 0, 'Participant 1', ha='center', fontdict={'fontsize':32})
    ax = fig.add_subplot(grid[id_grid+5])
    ax.axis('off')
    ax.text(0.5, 0, 'Participant 2', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid+6])
    ax.axis('off')
    ax.text(0.5, 0, 'Participant 3', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid+7])
    ax.axis('off')
    ax.text(0.5, 0, 'Participant 4', ha='center', fontdict={'fontsize':30})

ax = fig.add_subplot(grid[4])
ax.axis('off')
ax.text(0.9, 0.5, 'LH', ha='center', rotation=270, fontdict={'fontsize':30})

ax = fig.add_subplot(grid[9])
ax.axis('off')
ax.text(0.9, 0.5, 'RH', ha='center', rotation=270, fontdict={'fontsize':30})

# colorbar
cax = plt.axes([1.02, 0.32, 0.03, 0.4])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.set_ticks([])
cbar.ax.set_title('max', fontdict={'fontsize':30}, pad=20)
cbar.ax.set_xlabel('min', fontdict={'fontsize':30}, labelpad=20)

plt.show()

# =============================================================================
# #### GET FUNCTIONAL GRADIENTS EP ####
# =============================================================================

Vn = 2

fontcolor='black'
    
stri_file = f'{codeFolder}/masks/striatum_2mm.nii'
stri_msk = image.load_img(stri_file)
stri_msk_cropped = image.load_img(f'{codeFolder}/masks/striatum_cropped.nii')
stri_side = 'left'
    
eig_file = f'{resultFolder}/tasks/{task}/cohorts/{grp}/Vn{Vn}_eigenvector.nii'
# img_file = f'{resultFolder}/projection/boot/{task}/1_eigenvector_projection_{hip[:-4]}_fix.nii'
# eig_file = f'{resultFolder}/projection/boot/{task}/1_Vn{Vn}_eigenvector.nii'
eig = image.load_img(eig_file)

eig_1d = masking.apply_mask(eig, stri_msk_cropped)
eig_mni = masking.unmask(eig_1d, stri_msk)
eig_stri_1d = masking.apply_mask(eig_mni, stri_msk)
eig_stri = masking.unmask(eig_stri_1d, stri_msk)

vmin = min(eig_stri_1d)
vmax = max(eig_stri_1d)

codeFolder = '/Volumes/Scratch/functional_integration_psychosis/code/subcortex'
resultFolder = f'{codeFolder}/result/HCP-EP-striatum'
grp = 'P'
grp_folder = 'cohorts'
task = 'resting-state'

neuroshapeFolder = '/Volumes/Scratch/functional_integration_psychosis/code/neuroshape'

interpolation='linear'
kind='ball'
radius=3
n_samples = None
mask_img = f'{codeFolder}/masks/cortex.nii'
row = 0
cmap = cmap

P_Phi_lh = np.zeros((105, 32492))
P_Phi_rh = np.zeros((105, 32492))

text_file = f'{baseFolder}/{grp_folder}/{grp}/subjects.txt'
subjects = np.loadtxt(text_file, dtype='str').squeeze()

i = 0
for subject in subjects:
    img_file = f'{resultFolder}/tasks/{task}/{grp_folder}/{grp}/sub-{subject}_Vn{Vn}_eigenvector_projection_striatum_2mm.nii'
    img = image.load_img(img_file)
    
    func_surf_fslr = transforms.mni152_to_fslr(img, '32k')
    func_surf_lh = func_surf_fslr[0].agg_data()
    func_surf_rh = func_surf_fslr[1].agg_data()
    
    # save to array
    P_Phi_lh[i] = func_surf_lh
    P_Phi_rh[i] = func_surf_rh
        
    i += 1
        
texture_lh = P_Phi_lh
texture_rh = P_Phi_rh

grp = 'EP'

# create figure
fig = plt.figure(figsize=(30,9), constrained_layout=False)
grid = gridspec.GridSpec(
    2, 5, left=0., right=1., bottom=0., top=1.,
    height_ratios=[1.,1.], width_ratios=[0.6,1.2,1.2,1.2,1.2],
    hspace=0.0, wspace=0.0)

view='lateral'

print(f'vmin={vmin}, vmax={vmax}')

texture_stri = surface.vol_to_surf(
    eig, mesh, interpolation='linear', radius=3)#, mask_img=mask_file)

for sub in [0, 1, 2, 3]:
    for hemi in ['left', 'right']:
        if hemi == 'left':
            pial_mesh = f'{neuroshapeFolder}/data/fsaverage.L.pial_orig.32k_fs_LR.surf.gii'
            
            texture = texture_lh[sub]
            
            ax = fig.add_subplot(grid[sub+1], projection='3d')
            plotting.plot_surf(
                pial_mesh, texture, hemi=hemi, view=view,
                colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
                axes=ax, avg_method='mean')
            ax.dist = 7
        
            # plot striatum
            # ax = fig.add_subplot(grid[0])
            # disp = plotting.plot_img(
            #     eig_stri, display_mode='z', threshold=0, cmap=cmap, 
            #     vmin=vmin_stri, vmax=vmax_stri,
            #     axes=ax, cut_coords=cut_coords, colorbar=False, annotate=False)
            # disp.annotate(size=25)
            # texture_stri = surface.vol_to_surf(
            #     eig, mesh, interpolation='nearest', radius=3, mask_img=mask_file)
        
            # plot
            ax = fig.add_subplot(grid[0], projection='3d')
            plotting.plot_surf(mesh, texture_stri, view='dorsal', vmin=0, vmax=max(texture_stri),
                                cmap=cmap, avg_method='mean',
                                axes=ax)
            
            ax.text(-20,-35, 15, 'L', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            ax.text(20, -35, 15, 'R', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            #ax.text(100, 0, 0, text, ha='center', fontdict={'fontsize':30})
            ax.text(40, -18, 0, 'dorsal view', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            
        else:
            pial_mesh = f'{neuroshapeFolder}/data/fsaverage.R.pial_orig.32k_fs_LR.surf.gii'
            
            texture = texture_rh[sub]
            
            ax = fig.add_subplot(grid[sub+6], projection='3d')
            plotting.plot_surf(
                pial_mesh, texture, hemi=hemi, view=view,
                colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
                axes=ax, avg_method='mean')
            ax.dist = 7
            
            # plot striatum
            # ax = fig.add_subplot(grid[3])
            # disp = plotting.plot_img(
            #     eig_stri, display_mode='y', threshold=0, cmap=cmap, 
            #     vmin=vmin, vmax=vmax,
            #     axes=ax, cut_coords=cut_coords, colorbar=False, annotate=False)
            # disp.annotate(size=25)
            #texture_stri = surface.vol_to_surf(
                #eig, mesh, interpolation='nearest', radius=3, mask_img=mask_file)
        
            # plot
            ax = fig.add_subplot(grid[5], projection='3d')
            plotting.plot_surf(mesh, texture_stri, view='ventral', vmin=0, vmax=max(texture_stri),
                                cmap=cmap, avg_method='mean',
                                axes=ax)
            ax.text(-20, -35, 15, 'L', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            ax.text(20, -35, 15, 'R', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            #ax.text(-40, 0, 0, text, ha='center', fontdict={'fontsize':30})
            ax.text(-40, -18, 0, 'ventral view', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            #ax.set_facecolor('lightslategrey') #(0.4,0.4,0.4))
        
    
for label, id_grid in zip([f'Gradient {Vn-1} : {task} - {grp}'], [2]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')
    ax.text(0.8, 1, label, ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid+4])
    ax.axis('off')
    ax.text(0.5, 0, 'Participant 1', ha='center', fontdict={'fontsize':32})
    ax = fig.add_subplot(grid[id_grid+5])
    ax.axis('off')
    ax.text(0.5, 0, 'Participant 2', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid+6])
    ax.axis('off')
    ax.text(0.5, 0, 'Participant 3', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid+7])
    ax.axis('off')
    ax.text(0.5, 0, 'Participant 4', ha='center', fontdict={'fontsize':30})

ax = fig.add_subplot(grid[4])
ax.axis('off')
ax.text(0.9, 0.5, 'LH', ha='center', rotation=270, fontdict={'fontsize':30})

ax = fig.add_subplot(grid[9])
ax.axis('off')
ax.text(0.9, 0.5, 'RH', ha='center', rotation=270, fontdict={'fontsize':30})

# colorbar
cax = plt.axes([1.02, 0.32, 0.03, 0.4])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.set_ticks([])
cbar.ax.set_title('max', fontdict={'fontsize':30}, pad=20)
cbar.ax.set_xlabel('min', fontdict={'fontsize':30}, labelpad=20)

plt.show()
