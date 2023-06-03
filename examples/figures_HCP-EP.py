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
import nibabel as nib


##############################
# HIPPOCAMPUS/SUBCORTEX MESH #
##############################

# create point cloud
# cd /home/leonie/Documents/source/gradientography-task-fmri
codeFolder = '/Volumes/Scratch/functional_integration_psychosis/code/subcortex'
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

fontcolor = 'black'
resultFolder = 'result/HCP-EP-striatum'
# resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_nimg'

cmap = plt.get_cmap('viridis')
grp_folder = 'cohorts'
grps = ['HC','P']
grps_label = ['Healthy Cohort (HC)', 'Early Psychosis (EP)']

fig = plt.figure(figsize=(25,15), constrained_layout=False)
grid = gridspec.GridSpec(
    2, 6, left=0., right=1., bottom=0., top=0.8,
    height_ratios=[1,1], width_ratios=[1,1,1,1,1,1],
    hspace=0.0, wspace=0.0)

i = 0
task = 'resting-state'
img = image.load_img('masks/subcortex_mask.nii')
stri_file = 'masks/striatum_2mm.nii'
stri_msk = image.load_img(stri_file)
sub_msk = image.load_img('masks/subcortex_mask.nii')


for grp in grps:
    for Vn in [2, 3, 4]:
        # texture
        eig_file = f'{resultFolder}/tasks/{task}/{grp_folder}/{grp}/Vn{Vn}_eigenvector.nii'
        eig = image.load_img(eig_file)
        
        
        # eig_resample = image.resample_img(eig, target_affine=img.affine, target_shape=img.shape,
        #                                    interpolation='linear')

        # eig_1d = masking.apply_mask(eig_resample, sub_msk)
        # eig_mni = masking.unmask(eig_1d, sub_msk)
        # eig_stri_1d = masking.apply_mask(eig_mni, stri_msk)
        # eig_stri = masking.unmask(eig_stri_1d, stri_msk)
        # nib.save(eig_stri, f'{resultFolder}/tasks/{task}/{grp_folder}/{grp}/Vn{Vn}_eigenvector_resampled.nii')
        
        #img_1d = masking.apply_mask(img, 'masks/GMmask.nii')
        # vmin = min(eig_stri_1d)
        # vmax = max(eig_stri_1d)
        texture = surface.vol_to_surf(
            eig, mesh, interpolation='linear', radius=5, mask_img=mask_file)
        
        # plot
        for view in ['dorsal', 'ventral']:
            # plot
            
            if view == 'dorsal':
                ax = fig.add_subplot(grid[i], projection='3d')
                plotting.plot_surf(mesh, texture, view=view, vmin=0, vmax=max(texture),
                                    cmap=cmap, avg_method='mean',
                                    axes=ax)
                
                ax.text(37, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':fontcolor})
                ax.text(-35, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':fontcolor})
                # if task == 'resting_state':
                #     ax.text(40, -15, 53, 'dorsal view', va='center', fontdict={'fontsize':30, 'color':fontcolor})
                
            else:
                ax = fig.add_subplot(grid[i+6], projection='3d')
                plotting.plot_surf(mesh, texture, view=view, vmin=0, vmax=max(texture),
                                    cmap=cmap, avg_method='mean',
                                    axes=ax)
                ax.text(38, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':'white'})
                ax.text(-36, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':'white'})
                #ax.text(40, -15, 53, 'ventral view', va='center', fontdict={'fontsize':30, 'color':'white'})
                # variance explained
                
                ax.set_facecolor('lightslategrey') #(0.4,0.4,0.4))
        i += 1
    #i += 1

# add text
# for label, id_grid in zip(['RESTING-STATE'], [5]):
#     ax = fig.add_subplot(grid[id_grid])
#     ax.axis('off')    
#     ax.text(0.3, 0.5, label, rotation=90, 
#             va='center', fontdict={'fontsize':30})

for label, id_grid in zip(['dorsal view','ventral view'], [0, 6]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')
    ax.text(-0.20, 0.5, label, va='center', rotation=-90, fontdict={'fontsize':30, 'color':fontcolor})


#ax.text(45, 0, 50, f'{var.loc[Vn-2,0]:.2f}%', ha='center', fontdict={'fontsize':30})

for label, id_grid in zip(grps_label, [6, 9]):
    if id_grid == 6:
        var = pd.read_csv(f'{resultFolder}/tasks/{task}/{grp_folder}/HC/variance_explained.csv',
                      header=None)
    else:
        var = pd.read_csv(f'{resultFolder}/tasks/{task}/{grp_folder}/P/variance_explained.csv',
                      header=None)
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')
    ax.text(1.5, 2, label, ha='center', fontdict={'fontsize':30})
    ax.text(0.5, -.05, f'Gradient I:\n {var.loc[0,0]:.2f}%', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid+1])
    ax.axis('off')
    ax.text(0.5, -.05, f'Gradient II:\n {var.loc[1,0]:.2f}%', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid+2])
    ax.axis('off')
    ax.text(0.5, -.05, f'Gradient III:\n {var.loc[2,0]:.2f}%', ha='center', fontdict={'fontsize':30})

# colorbar
cax = plt.axes([1.01, 0.3, 0.03, 0.3])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.set_ticks([])
cbar.ax.set_title('max', fontdict={'fontsize':30}, pad=20)
cbar.ax.set_xlabel('0', fontdict={'fontsize':30}, labelpad=20)

plt.show()

################################
# FIGURE. 8 GRADIENTS EIGENMAP #
################################

fontcolor = 'black'
resultFolder = 'result/HCP-EP-striatum'
# resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_nimg'

cmap = plt.get_cmap('viridis')
grp_folder = 'cohorts'
#grps = ['HC','P']
#grps_label = ['Healthy Cohort (HC)', 'Early Psychosis (EP)']

fig = plt.figure(figsize=(28,11), constrained_layout=False)
grid = gridspec.GridSpec(
    2, 8, left=0., right=1., bottom=0., top=0.9,
    height_ratios=[1,1], width_ratios=[1,1,1,1,1,1,1,1],
    hspace=0.0, wspace=0.0)

i = 0
task = 'resting-state'
img = image.load_img('masks/subcortex_mask.nii')
stri_file = 'masks/striatum_2mm.nii'
stri_msk = image.load_img(stri_file)
sub_msk = image.load_img('masks/subcortex_mask.nii')

grp = 'HC'

for Vn in [2, 3, 4, 5, 6, 7, 8, 9]:
        # texture
    eig_file = f'{resultFolder}/tasks/{task}/{grp_folder}/{grp}/Vn{Vn}_eigenvector.nii'
    eig = image.load_img(eig_file)
    
    
    eig_resample = image.resample_img(eig, target_affine=img.affine, target_shape=img.shape,
                                       interpolation='linear')

    eig_1d = masking.apply_mask(eig_resample, sub_msk)
    eig_mni = masking.unmask(eig_1d, sub_msk)
    eig_stri_1d = masking.apply_mask(eig_mni, stri_msk)
    eig_stri = masking.unmask(eig_stri_1d, stri_msk)
    nib.save(eig_stri, f'{resultFolder}/tasks/{task}/{grp_folder}/{grp}/Vn{Vn}_eigenvector_resampled.nii')
    
    #img_1d = masking.apply_mask(img, 'masks/GMmask.nii')
    vmin = min(eig_stri_1d)
    vmax = max(eig_stri_1d)
    texture = surface.vol_to_surf(
        eig_stri, mesh, interpolation='linear', radius=5, mask_img=mask_file)
    
    # plot
    for view in ['dorsal', 'ventral']:
        # plot
        
        if view == 'dorsal':
            ax = fig.add_subplot(grid[i], projection='3d')
            plotting.plot_surf(mesh, texture, view=view, vmin=0, vmax=max(texture),
                                cmap=cmap, avg_method='mean',
                                axes=ax)
            
            ax.text(37, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            ax.text(-35, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            if task == 'resting_state':
                ax.text(40, -15, 53, 'dorsal view', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            
        else:
            ax = fig.add_subplot(grid[i+8], projection='3d')
            plotting.plot_surf(mesh, texture, view=view, vmin=0, vmax=max(texture),
                                cmap=cmap, avg_method='mean',
                                axes=ax)
            ax.text(38, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':'white'})
            ax.text(-36, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':'white'})
            if task == 'resting_state':
                ax.text(0, 0, 0, 'ventral view', va='center', fontdict={'fontsize':30, 'color':'white'})
            ax.set_facecolor('lightslategrey') #(0.4,0.4,0.4))
    i += 1
    #i += 1

# add text
# for label, id_grid in zip(['RESTING-STATE'], [5]):
#     ax = fig.add_subplot(grid[id_grid])
#     ax.axis('off')    
#     ax.text(0.3, 0.5, label, rotation=90, 
#             va='center', fontdict={'fontsize':30})


ax = fig.add_subplot(grid[8])
ax.axis('off')
#ax.text(1.5, 2, label, ha='center', fontdict={'fontsize':30})
ax.text(0.5, .05, 'Gradient I', ha='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[9])
ax.axis('off')
ax.text(0.5, .05, 'Gradient II', ha='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[10])
ax.axis('off')
ax.text(0.5, .05, 'Gradient III', ha='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[11])
ax.axis('off')
#ax.text(1.5, 2, label, ha='center', fontdict={'fontsize':30})
ax.text(0.5, .05, 'Gradient IV', ha='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[12])
ax.axis('off')
ax.text(0.5, .05, 'Gradient V', ha='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[13])
ax.axis('off')
ax.text(0.5, .05, 'Gradient VI', ha='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[14])
ax.axis('off')
#ax.text(1.5, 2, label, ha='center', fontdict={'fontsize':30})
ax.text(0.5, .05, 'Gradient VII', ha='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[15])
ax.axis('off')
#ax.text(1.5, 2, label, ha='center', fontdict={'fontsize':30})
ax.text(0.5, .05, 'Gradient VIII', ha='center', fontdict={'fontsize':30})


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

resultFolder = 'result/HCP-EP-striatum/'
# resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_nimg'
cmap = plt.get_cmap('viridis')
grp_folder = 'cohorts'
grps = ['HC', 'P']
grps_label = ['Healthy Cohort (HC)', 'Early Psychosis (EP)']

fontcolor = 'black'

fig = plt.figure(figsize=(25,12), constrained_layout=False)
grid = gridspec.GridSpec(
    2, 6, left=0., right=1., bottom=0., top=0.9,
    height_ratios=[1,1], width_ratios=[1,1,1,1,1,1],
    hspace=0.0, wspace=0.0)

i = 0

for task in ['resting-state']:
    for grp in grps:
        for Vn in [2, 3, 4]:
        # texture
        
            mag_file = f'{resultFolder}/tasks/{task}/{grp_folder}/{grp}/Vn{Vn}_magnitude.nii'
            # mag_file = f'{resultFolder}/tasks/{task}/cohort/Vn{Vn}_{grp}_mean_boot.nii'
            mag = image.load_img(mag_file)
            
            #eig_file = f'{resultFolder}/tasks/{task}/{grp_folder}/{grp}/Vn{Vn}_eigenvector.nii'
            #eig = image.load_img(eig_file)
            #texture = surface.vol_to_surf(
                #mag, mesh, interpolation='linear', radius=5, mask_img=mask_file)
            
            # mag_resample = image.resample_img(mag, target_affine=img.affine, target_shape=img.shape,
            #                                    interpolation='linear')
    
            # mag_1d = masking.apply_mask(mag_resample, sub_msk)
            # mag_mni = masking.unmask(mag_1d, sub_msk)
            # mag_stri_1d = masking.apply_mask(mag_mni, stri_msk)
            # mag_stri = masking.unmask(mag_stri_1d, stri_msk)
            #img_1d = masking.apply_mask(img, 'masks/GMmask.nii')
            
            texture = surface.vol_to_surf(
                mag, mesh, interpolation='nearest', radius=5, mask_img=mask_file)
            for view in ['dorsal', 'ventral']:
                # plot
                
                if view == 'dorsal':
                    ax = fig.add_subplot(grid[i], projection='3d')
                    plotting.plot_surf(mesh, texture, view=view, vmin=0, vmax=0.2,
                                        cmap=cmap, avg_method='mean',
                                        axes=ax)
                    ax.text(37, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':fontcolor})
                    ax.text(-35, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':fontcolor})
                    if task == 'resting_state':
                        ax.text(40, -15, 53, 'dorsal view', va='center', fontdict={'fontsize':30, 'color':fontcolor})
                else:
                    ax = fig.add_subplot(grid[i+6], projection='3d')
                    plotting.plot_surf(mesh, texture, view=view, vmin=0, vmax=0.2,
                                        cmap=cmap, avg_method='mean',
                                        axes=ax)
                    ax.text(38, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':'white'})
                    ax.text(-36, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':'white'})
                    if task == 'resting_state':
                        ax.text(0, 0, 0, 'ventral view', va='center', fontdict={'fontsize':30, 'color':'white'})
                    ax.set_facecolor('lightslategrey') #(0.4,0.4,0.4))
                
            i += 1

# add text

for label, id_grid in zip(grps_label, [6, 9]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')
    ax.text(1.5, 2, label, ha='center', fontdict={'fontsize':30})
    ax.text(0.5, .05, 'Gradient I', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid+1])
    ax.axis('off')
    ax.text(0.5, .05, 'Gradient II', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid+2])
    ax.axis('off')
    ax.text(0.5, .05, 'Gradient III', ha='center', fontdict={'fontsize':30})

for label, id_grid in zip(['dorsal view','ventral view'], [0, 6]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')
    ax.text(-0.20, 0.5, label, va='center', rotation=-90, fontdict={'fontsize':30, 'color':fontcolor})

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

resultFolder = f'{codeFolder}/result/HCP-EP-striatum'
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
grps = ['HC', 'P']
grps_label = ['HC', 'P']

fontcolor = 'darkslategrey'
fontcolor='black'

cmap=plt.get_cmap('twilight_shifted')
twil = cm.get_cmap('twilight_shifted', 1000)
newcolors = np.vstack((twil(np.linspace(0, 0.3, 400)),
                        twil(np.linspace(0.4, 0.6, 400)),
                        twil(np.linspace(0.7, 1, 400))))
cmap = ListedColormap(newcolors, name='twilight_shifted_threshold')

fig = plt.figure(figsize=(15,12), constrained_layout=False)
grid = gridspec.GridSpec(
    2, 3, left=0., right=1., bottom=0., top=0.9,
    height_ratios=[1,1], width_ratios=[1,1,1],
    hspace=0.0, wspace=0.0)

i = 0
task = 'resting-state'

for Vn in [2, 3, 4]:
    
     
    eig_file = f'{resultFolder}/tasks/{task}/{grp_folder}/Vn{Vn}_eig_z_{grps[1]}-{grps[0]}.nii'
    eig = image.load_img(eig_file)
    #eig_msk = f'{resultFolder}/tasks/{task}/{grp_folder}/Vn{Vn}_pval_thr.nii'
    #mag_1d = masking.apply_mask(mag, mag_msk)
    #mag_thr = masking.unmask(mag_1d, mag_msk)
    
    
    #img_1d = masking.apply_mask(img, 'masks/GMmask.nii')
    
    texture = surface.vol_to_surf(
        eig, mesh, interpolation='nearest', radius=3, mask_img=mask_file)
    
    # texture = surface.vol_to_surf(
    #     mag_thr, mesh, interpolation=interpolation, radius=radius, 
    #     kind=kind, mask_img=mask_file)
    
    vmax = 4.0
    for view in ['dorsal', 'ventral']:
        # plot
        
        if view == 'dorsal':
            ax = fig.add_subplot(grid[i], projection='3d')
            plotting.plot_surf(mesh, texture, view=view, vmin=-vmax, vmax=vmax,
                                cmap=cmap, avg_method=avg_method,
                                axes=ax)
            ax.text(37, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            ax.text(-35, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            #ax.text(40, -15, 53, 'dorsal view', va='center', fontdict={'fontsize':30, 'color':fontcolor})     
        else:
            ax = fig.add_subplot(grid[i+3], projection='3d')
            plotting.plot_surf(mesh, texture, view=view, vmin=-vmax, vmax=vmax,
                                cmap=cmap, avg_method=avg_method,
                                axes=ax)
            ax.text(38, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':'white'})
            ax.text(-38, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':'white'})
            ax.set_facecolor('lightslategrey')
            #ax.text(-40, -22, -38, 'ventral view', va='center', fontdict={'fontsize':30, 'color':'white'})
    i += 1

# add text
# for label, id_grid in zip(['GO', 'NOGO', 'BACKGROUND'], [3, 6, 9]):
#     ax = fig.add_subplot(grid[id_grid])
#     ax.axis('off')    
#     ax.text(0.3, 0.5, label, rotation=90, 
#             va='center', fontdict={'fontsize':30})

ax = fig.add_subplot(grid[0])
ax.axis('off')    
ax.text(1.5, 1, 'Healthy Controls (HC) - Early Psychosis (EP)', ha='center', 
        fontdict={'fontsize':30, 'color':fontcolor})


ax = fig.add_subplot(grid[3])
ax.axis('off')
ax.text(0.5, -0.05, 'Gradient I', ha='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[4])
ax.axis('off')
ax.text(0.5, -0.05, 'Gradient II', ha='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[5])
ax.axis('off')
ax.text(0.5, -0.05, 'Gradient III', ha='center', fontdict={'fontsize':30})

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

################################################
# FIGURE. COHORT COMPARISON THRESHOLDED P<0.05 #
################################################

resultFolder = f'{codeFolder}/result/HCP-EP-striatum'
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
grps = ['HC', 'P']
grps_label = ['HC', 'P']

fontcolor = 'darkslategrey'
fontcolor='black'

cmap=plt.get_cmap('twilight_shifted')
twil = cm.get_cmap('twilight_shifted', 1000)
newcolors = np.vstack((twil(np.linspace(0, 0.3, 400)),
                        twil(np.linspace(0.4, 0.6, 400)),
                        twil(np.linspace(0.7, 1, 400))))
cmap = ListedColormap(newcolors, name='twilight_shifted_threshold')

fig = plt.figure(figsize=(15,12), constrained_layout=False)
grid = gridspec.GridSpec(
    2, 3, left=0., right=1., bottom=0., top=0.9,
    height_ratios=[1,1], width_ratios=[1,1,1],
    hspace=0.0, wspace=0.0)

i = 0
task = 'resting-state'

for Vn in [2, 3, 4]:
    
     
    eig_file = f'{resultFolder}/tasks/{task}/{grp_folder}/Vn{Vn}_eig_z_{grps[1]}-{grps[0]}.nii'
    eig = image.load_img(eig_file)
    #eig_msk = f'{resultFolder}/tasks/{task}/{grp_folder}/Vn{Vn}_pval_thr.nii'
    #mag_1d = masking.apply_mask(mag, mag_msk)
    #mag_thr = masking.unmask(mag_1d, mag_msk)
    
    
    #img_1d = masking.apply_mask(img, 'masks/GMmask.nii')
    
    texture = surface.vol_to_surf(
        eig, mesh, interpolation='nearest', radius=3, mask_img=mask_file)
    
    # texture = surface.vol_to_surf(
    #     mag_thr, mesh, interpolation=interpolation, radius=radius, 
    #     kind=kind, mask_img=mask_file)
    
    vmax = 4.0
    for view in ['dorsal', 'ventral']:
        # plot
        
        if view == 'dorsal':
            ax = fig.add_subplot(grid[i], projection='3d')
            plotting.plot_surf(mesh, texture, view=view, vmin=-vmax, vmax=vmax,
                                cmap=cmap, avg_method=avg_method,
                                axes=ax)
            ax.text(37, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            ax.text(-35, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            #ax.text(40, -15, 53, 'dorsal view', va='center', fontdict={'fontsize':30, 'color':fontcolor})     
        else:
            ax = fig.add_subplot(grid[i+3], projection='3d')
            plotting.plot_surf(mesh, texture, view=view, vmin=-vmax, vmax=vmax,
                                cmap=cmap, avg_method=avg_method,
                                axes=ax)
            ax.text(38, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':'white'})
            ax.text(-38, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':'white'})
            ax.set_facecolor('lightslategrey')
            #ax.text(-40, -22, -38, 'ventral view', va='center', fontdict={'fontsize':30, 'color':'white'})
    i += 1

# add text
# for label, id_grid in zip(['GO', 'NOGO', 'BACKGROUND'], [3, 6, 9]):
#     ax = fig.add_subplot(grid[id_grid])
#     ax.axis('off')    
#     ax.text(0.3, 0.5, label, rotation=90, 
#             va='center', fontdict={'fontsize':30})

ax = fig.add_subplot(grid[0])
ax.axis('off')    
ax.text(1.5, 1, 'Healthy Controls (HC) - Early Psychosis (EP)', ha='center', 
        fontdict={'fontsize':30, 'color':fontcolor})


ax = fig.add_subplot(grid[3])
ax.axis('off')
ax.text(0.5, -0.05, 'Gradient I', ha='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[4])
ax.axis('off')
ax.text(0.5, -0.05, 'Gradient II', ha='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[5])
ax.axis('off')
ax.text(0.5, -0.05, 'Gradient III', ha='center', fontdict={'fontsize':30})

# colorbar
cax = plt.axes([1.07, 0.3, 0.05, 0.3])
cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax), cmap=cmap), cax=cax)
cbar.set_ticks([-2,2])
cbar.set_ticklabels([r'-2$\sigma$',r'2$\sigma$'])
cbar.ax.tick_params(labelsize=30, labelcolor=fontcolor)
cbar.ax.set_title(f'{grps_label[1]}>{grps_label[0]}', fontdict={'fontsize':30, 'color':fontcolor}, pad=20)
cbar.ax.set_xlabel(f'{grps_label[0]}>{grps_label[1]}', fontdict={'fontsize':30, 'color':fontcolor}, labelpad=20)

plt.show()

###############################################
# FIGURE. CORTICAL PROJECTION - RESTING STATE #
###############################################

# parameters
resultFolder = 'result/HCP-EP-striatum'
# resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_age_match' ###

cmap = plt.get_cmap('viridis')
Vn = 2
grp = 'HC' ###
task = 'resting-state' ###
hemi = 'left'
#mask_file = 'masks/striatum_cropped.nii'
    
stri_file = 'masks/striatum_2mm.nii'
stri_msk = image.load_img(stri_file)
stri_msk_cropped = image.load_img('masks/striatum_cropped.nii')
stri_side = 'left'
    
img_file = f'{resultFolder}/tasks/{task}/cohorts/{grp}/Vn{Vn}_eigenvector_projection_striatum_2mm.nii'
eig_file = f'{resultFolder}/tasks/{task}/cohorts/{grp}/Vn{Vn}_eigenvector.nii'
# img_file = f'{resultFolder}/projection/boot/{task}/1_eigenvector_projection_{hip[:-4]}_fix.nii'
# eig_file = f'{resultFolder}/projection/boot/{task}/1_Vn{Vn}_eigenvector.nii'
img = image.load_img(img_file)
eig = image.load_img(eig_file)

eig_1d = masking.apply_mask(eig, stri_msk_cropped)
eig_mni = masking.unmask(eig_1d, stri_msk)
eig_stri_1d = masking.apply_mask(eig_mni, stri_msk)
eig_stri = masking.unmask(eig_stri_1d, stri_msk)

#img_1d = masking.apply_mask(img, 'masks/GMmask.nii')
vmin = min(eig_stri_1d)
vmax = max(eig_stri_1d)
# vmin_stri = min(eig_stri_1d)
# vmax_stri = max(eig_stri_1d)

#fsaverage = datasets.fetch_surf_fsaverage()

# create figure
fig = plt.figure(figsize=(15,9), constrained_layout=False)
grid = gridspec.GridSpec(
    2, 3, left=0., right=1., bottom=0., top=1.,
    height_ratios=[1.,1.], width_ratios=[0.6,1,0.6],
    hspace=0.0, wspace=0.0)

interpolation='linear'
kind='ball'
radius=3
n_samples = None
mask_img = 'masks/cortex.nii'
row = 0
cmap = cmap
# cmap = plt.get_cmap('bone_r')

print(f'vmin={vmin}, vmax={vmax}')

texture_stri = surface.vol_to_surf(
    eig, mesh, interpolation='nearest', radius=3)#, mask_img=mask_file)


for hemi in ['left', 'right']:
    if hemi == 'left':
        pial_mesh = '/Volumes/Scratch/functional_integration_psychosis/code/subcortex/meshes/1001_01_MR.L.pial.32k_fs_LR.surf.gii'#fsaverage.pial_left
        infl_mesh = '/Volumes/Scratch/functional_integration_psychosis/code/subcortex/meshes/1001_01_MR.L.inflated.32k_fs_LR.surf.gii'#fsaverage.infl_left
        cut_coords=[-2]
        
        texture = surface.vol_to_surf(
            img, pial_mesh, interpolation=interpolation, 
            mask_img=mask_img, kind=kind, radius=radius, n_samples=n_samples)
        r = 30
        new_text = surface.vol_to_surf(
            img, pial_mesh, interpolation=interpolation, 
            mask_img=mask_img, kind=kind, radius=r, n_samples=n_samples)
        texture[texture != texture] = new_text[texture != texture]
        
        for col in [1, 2]:
            if col == 1:
                view = 'lateral'
            else:
                view = 'medial'
            # ax = fig.add_subplot(2, 3, row*3+col+1, projection='3d')
            ax = fig.add_subplot(grid[col], projection='3d')
            plotting.plot_surf(
                infl_mesh, texture, hemi=hemi, view=view,
                colorbar=False, cmap=cmap, vmin=0, vmax=vmax,
                axes=ax)
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
        ax.text(40, -15, 0, 'dorsal view', va='center', fontdict={'fontsize':30, 'color':fontcolor})
        
    else:
        pial_mesh = '/Volumes/Scratch/functional_integration_psychosis/code/subcortex/meshes/1001_01_MR.R.pial.32k_fs_LR.surf.gii'#fsaverage.pial_left
        infl_mesh = '/Volumes/Scratch/functional_integration_psychosis/code/subcortex/meshes/1001_01_MR.R.inflated.32k_fs_LR.surf.gii'
        
        cut_coords=[10]
        
        texture = surface.vol_to_surf(
            img, pial_mesh, interpolation=interpolation, 
            mask_img=mask_img, kind=kind, radius=radius, n_samples=n_samples)
        r = 30
        new_text = surface.vol_to_surf(
            img, pial_mesh, interpolation=interpolation, 
            mask_img=mask_img, kind=kind, radius=r, n_samples=n_samples)
        texture[texture != texture] = new_text[texture != texture]
        
        for col in [1, 2]:
            if col == 1:
                view = 'lateral'
            else:
                view = 'medial'
            # ax = fig.add_subplot(2, 3, row*3+col+1, projection='3d')
            ax = fig.add_subplot(grid[col+3], projection='3d')
            plotting.plot_surf(
                infl_mesh, texture, hemi=hemi, view=view,
                colorbar=False, cmap=cmap, vmin=0, vmax=vmax,
                axes=ax)
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
        ax = fig.add_subplot(grid[3], projection='3d')
        plotting.plot_surf(mesh, texture_stri, view='ventral', vmin=0, vmax=max(texture_stri),
                            cmap=cmap, avg_method='mean',
                            axes=ax)
        ax.text(-20, -35, 15, 'L', va='center', fontdict={'fontsize':30, 'color':fontcolor})
        ax.text(20, -35, 15, 'R', va='center', fontdict={'fontsize':30, 'color':fontcolor})
        #ax.text(-40, 0, 0, text, ha='center', fontdict={'fontsize':30})
        ax.text(-40, -15, 0, 'ventral view', va='center', fontdict={'fontsize':30, 'color':fontcolor})
        #ax.set_facecolor('lightslategrey') #(0.4,0.4,0.4))
        
    
for label, id_grid in zip([f'Gradient {Vn-1} : {task} - {grp}'], [1, 6]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')
    ax.text(0.5, 1, label, ha='center', fontdict={'fontsize':30})
    ax.text(0.9, 0, 'Left hemisphere', ha='center', fontdict={'fontsize':32})
    ax = fig.add_subplot(grid[id_grid+3])
    ax.axis('off')
    ax.text(0.9, 0, 'Right hemisphere', ha='center', fontdict={'fontsize':30})
    
# colorbar
cax = plt.axes([1.05, 0.32, 0.03, 0.4])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.set_ticks([])
cbar.ax.set_title('max', fontdict={'fontsize':30}, pad=20)
cbar.ax.set_xlabel('min', fontdict={'fontsize':30}, labelpad=20)

plt.show()

fig.savefig(f'/tmp/cortical_projection_{Vn}.png', bbox_inches='tight', dpi=300)
