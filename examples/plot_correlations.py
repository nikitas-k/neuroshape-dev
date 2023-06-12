import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from nilearn import image
from neuromaps import transforms
from neuromaps.datasets.atlases import fetch_atlas
from neuromaps.stats import compare_images
from nilearn import masking

fslr = fetch_atlas('fsLR', density='32k')

# =============================================================================
# #### Striatum gradients ####
# =============================================================================

codeFolder = '/Volumes/Scratch/functional_integration_psychosis/code/subcortex'
resultFolder = f'{codeFolder}/result/HCP-EP-striatum'

mask_file = f'{codeFolder}/masks/striatum_cropped.nii'
stri_file = f'{codeFolder}/masks/striatum_2mm.nii'
stri_lh = f'{codeFolder}/masks/striatum_2mm_lh.nii.gz'
stri_rh = f'{codeFolder}/masks/striatum_2mm_rh.nii.gz'
stri_img = image.load_img(stri_file)
stri_affine = stri_img.affine

task = 'resting-state'
grps = ['HC', 'P']

corr_stri_lh = np.zeros((20, 20))
corr_stri_rh = np.zeros((20, 20))

func_stri_HC = np.zeros((20, 3078))

func_stri_P = np.zeros((20, 3078))

i = 0
for Vn in range(2, 22):
    for grp in grps:
        
        eig = f'{resultFolder}/tasks/{task}/cohorts/{grp}/Vn{Vn}_eigenvector.nii'
        eig_1d = masking.apply_mask(eig, mask_file)
        eig_mni = masking.unmask(eig_1d, stri_file)
        eig_stri_1d = masking.apply_mask(eig_mni, stri_file)
        
        if grp == 'HC':
            func_stri_HC[i] = eig_stri_1d
            
        else:
            func_stri_P[i] = eig_stri_1d
    i += 1

    
Phi_HC = func_stri_HC
Phi_P = func_stri_P

Phi_lh_HC = Phi_HC[:,:Phi_HC.shape[1]//2]
Phi_rh_HC = Phi_HC[:,Phi_HC.shape[1]//2:]

# for g in range(Phi_HC.shape[0]):
#     Phi_lh_HC[g] = (Phi_lh_HC[g]-Phi_lh_HC[g].mean())/(Phi_lh_HC[g].std())
#     Phi_rh_HC[g] = (Phi_rh_HC[g]-Phi_rh_HC[g].mean())/(Phi_rh_HC[g].std())

Phi_lh_P = Phi_P[:,:Phi_P.shape[1]//2]
Phi_rh_P = Phi_P[:,Phi_P.shape[1]//2:]

# for g in range(Phi_P.shape[0]):
#     Phi_lh_P[g] = (Phi_lh_P[g]-Phi_lh_P[g].mean())/(Phi_lh_P[g].std())
#     Phi_rh_P[g] = (Phi_rh_P[g]-Phi_rh_P[g].mean())/(Phi_rh_P[g].std())

# =============================================================================
# #### Between-group gradient-to-gradient ####
# =============================================================================

corr_lh_phi_groups = np.zeros((Phi_lh_HC.shape[0],Phi_lh_HC.shape[0]))
corr_rh_phi_groups = np.zeros((Phi_rh_HC.shape[0],Phi_lh_HC.shape[0]))

for i in range(Phi_lh_HC.shape[0]):
    grad_lh_HC = Phi_lh_HC[i]
    grad_rh_HC = Phi_rh_HC[i]
    for j in range(Phi_lh_P.shape[0]):
        grad_lh_P = Phi_lh_P[j]
        grad_rh_P = Phi_rh_P[j]
        corr_lh = np.corrcoef(grad_lh_HC, grad_lh_P)[0][1]
        corr_rh = np.corrcoef(grad_rh_HC, grad_rh_P)[0][1]
        if corr_lh < 0:
            corr_lh = np.corrcoef(grad_lh_HC, -grad_lh_P)[0][1]
        if corr_rh < 0:
            corr_rh = np.corrcoef(grad_rh_HC, -grad_rh_P)[0][1]
        corr_lh_phi_groups[i,j] = corr_lh
        corr_rh_phi_groups[i,j] = corr_rh
        
cmap = plt.get_cmap('Reds')

corr_phi_groups = np.mean((corr_lh_phi_groups, corr_rh_phi_groups), axis=0)
        
# plot correlation
fig = plt.figure(figsize=(15,14), constrained_layout=False)
ax = sns.heatmap(corr_phi_groups, annot=False, annot_kws={'size':20},
                 cbar=False, 
                 cmap=cmap, vmin=0.0, vmax=1.)
ax.set(ylabel="")
ax_labels = [f'Gradient {Vn}' for Vn in range(1, 21)]
ax.set_xticklabels(labels=ax_labels, fontdict={'fontsize':18})
labels = ax.get_yticklabels
ax.set_yticklabels(labels=ax_labels, fontdict={'fontsize':18})
ax.margins(0.8)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title(r'Correlation of HC striatal $\phi_f$ to EP striatal $\phi_f$', fontdict={'fontsize':30})
cax = plt.axes([0.95, 0.15, 0.03, 0.7])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.ax.tick_params(labelsize=15)
cbar.set_label(label='Pearson correlation coefficient', fontdict={'fontsize':30})
plt.show()

# =============================================================================
# #### Striatum eigenmodes ####
# =============================================================================

lboFolder = '/Volumes/Scratch/functional_integration_psychosis/'
# TODO


# =============================================================================
# #### Cortical gradients ####
# =============================================================================

codeFolder = '/Volumes/Scratch/functional_integration_psychosis/code/subcortex'
resultFolder = f'{codeFolder}/result/HCP-EP-striatum'

i = 0
texture_func_HC = np.zeros((20, 64984))
for Vn in range(2, 22):
    grp = 'HC'
    task = 'resting-state'
    
    img_file = f'{resultFolder}/tasks/{task}/cohorts/{grp}/Vn{Vn}_eigenvector_projection_striatum_2mm.nii'
    img = image.load_img(img_file)
    
    
    func_surf_fslr = transforms.mni152_to_fslr(img, '32k')
    
    texture_func = np.hstack((func_surf_fslr[0].agg_data(), func_surf_fslr[1].agg_data()))
    
    texture_func_HC[i] = texture_func
    i += 1

texture_func_P = np.zeros((20, 64984))
i = 0
for Vn in range(2, 22):
    grp = 'P'
    task = 'resting-state'
    
    img_file = f'{resultFolder}/tasks/{task}/cohorts/{grp}/Vn{Vn}_eigenvector_projection_striatum_2mm.nii'
    img = image.load_img(img_file)
    
    
    func_surf_fslr = transforms.mni152_to_fslr(img, '32k')
    
    texture_func = np.hstack((func_surf_fslr[0].agg_data(), func_surf_fslr[1].agg_data()))
    
    texture_func_P[i] = texture_func
    
    i += 1
    
Phi_HC = texture_func_HC
Phi_P = texture_func_P

Phi_lh_HC = Phi_HC[:,:Phi_HC.shape[1]//2]
Phi_rh_HC = Phi_HC[:,Phi_HC.shape[1]//2:]

# for g in range(Phi_HC.shape[0]):
#     Phi_lh_HC[g] = (Phi_lh_HC[g]-Phi_lh_HC[g].mean())/(Phi_lh_HC[g].std())
#     Phi_rh_HC[g] = (Phi_rh_HC[g]-Phi_rh_HC[g].mean())/(Phi_rh_HC[g].std())

Phi_lh_P = Phi_P[:,:Phi_P.shape[1]//2]
Phi_rh_P = Phi_P[:,Phi_P.shape[1]//2:]

# for g in range(Phi_P.shape[0]):
#     Phi_lh_P[g] = (Phi_lh_P[g]-Phi_lh_P[g].mean())/(Phi_lh_P[g].std())
#     Phi_rh_P[g] = (Phi_rh_P[g]-Phi_rh_P[g].mean())/(Phi_rh_P[g].std())


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

Psi_HC = HC_ev_subwise_norm
Psi_lh_HC = Psi_HC[:,:,:Psi_HC.shape[2]//2]
Psi_rh_HC = Psi_HC[:,:,Psi_HC.shape[2]//2:]

#renormalize
# for sub in range(Psi_HC.shape[0]):
#     for j in range(Psi_HC.shape[1]):
#         Psi_lh_HC[sub,j] = (Psi_lh_HC[sub,j]-Psi_lh_HC[sub,j].mean())/(Psi_lh_HC[sub,j].std())
#         Psi_rh_HC[sub,j] = (Psi_rh_HC[sub,j]-Psi_rh_HC[sub,j].mean())/(Psi_rh_HC[sub,j].std())
        
Psi_P = P_ev_subwise_norm
Psi_lh_P = Psi_P[:,:,:Psi_P.shape[2]//2]
Psi_rh_P = Psi_P[:,:,Psi_P.shape[2]//2:]

#renormalize
# for sub in range(Psi_P.shape[0]):
#     for j in range(Psi_P.shape[1]):
#         Psi_lh_P[sub,j] = (Psi_lh_P[sub,j]-Psi_lh_P[sub,j].mean())/(Psi_lh_P[sub,j].std())
#         Psi_rh_P[sub,j] = (Psi_rh_P[sub,j]-Psi_rh_P[sub,j].mean())/(Psi_rh_P[sub,j].std())

# =============================================================================
# #### Within group mode-to-gradient correlation HC ####
# =============================================================================

corr_matrix_lh = np.zeros((Phi_lh_HC.shape[0], Phi_lh_HC.shape[0]))
corr_matrix_rh = np.zeros((Phi_lh_HC.shape[0], Phi_lh_HC.shape[0]))

for i in range(Phi_lh_HC.shape[0]):
    for j in range(Phi_lh_HC.shape[0]):
        corr_lh = np.zeros(Psi_lh_HC.shape[0])
        corr_rh = np.zeros(Psi_lh_HC.shape[0])
        
        for sub in range(Psi_lh_HC.shape[0]):
            rho = compare_images(Psi_lh_HC[sub,j], Phi_lh_HC[i])
            if rho < 0:
                rho = compare_images(Psi_lh_HC[sub,j], -Phi_lh_HC[i])
            
            corr_lh[sub] = rho
            
            rho = compare_images(Psi_rh_HC[sub,j], Phi_rh_HC[i])
            if rho < 0:
                rho = compare_images(Psi_rh_HC[sub,j], -Phi_rh_HC[i])
                
            corr_rh[sub] = rho
        
        corr_matrix_lh[i,j] = np.mean(corr_lh)
        corr_matrix_rh[i,j] = np.mean(corr_rh)
        
corr_matrix = np.mean((corr_matrix_lh, corr_matrix_rh), axis=0)
        
#matrix = np.triu(corr_matrix_lh, 1)

cmap = plt.get_cmap('Reds')

fig = plt.figure(figsize=(12,10), constrained_layout=False)
ax = sns.heatmap(corr_matrix, annot=False, annot_kws={'size':20},
                 cbar=False, 
                 cmap=cmap, vmin=0.0, vmax=1.)
ax.set(ylabel="")

y_labels = [f'Gradient {Vn}' for Vn in range(1, 21)]
x_labels = [f'Mode {J}' for J in range(1,21)]

ax.set_xticklabels(labels=x_labels, fontdict={'fontsize':18})
ax.set_yticklabels(labels=y_labels, fontdict={'fontsize':18})
ax.margins(0.8)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title(r'Correlation of HC gradient $\phi_f$ to HC mode $\psi_j$', fontdict={'fontsize':30})
cax = plt.axes([0.95, 0.15, 0.03, 0.7])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.ax.tick_params(labelsize=15)
cbar.set_label(label='Pearson correlation r', fontdict={'fontsize':20})
plt.show()

# =============================================================================
# #### Within group mode-to-gradient correlation EP ####
# =============================================================================

corr_matrix_lh = np.zeros((Phi_lh_P.shape[0], Phi_lh_P.shape[0]))
corr_matrix_rh = np.zeros((Phi_lh_P.shape[0], Phi_lh_P.shape[0]))

for i in range(Phi_lh_P.shape[0]):
    for j in range(Phi_lh_P.shape[0]):
        corr_lh = np.zeros(Phi_lh_P.shape[0])
        corr_rh = np.zeros(Phi_lh_P.shape[0])
        
        for sub in range(Phi_lh_P.shape[0]):
            rho = compare_images(Phi_lh_P[i], Psi_lh_P[sub,j])
            if rho < 0:
                rho = compare_images(Phi_lh_P[i], -Psi_lh_P[sub,j])
            
            corr_lh[sub] = rho
            
            rho = compare_images(Phi_rh_P[i], Psi_rh_P[sub,j])
            if rho < 0:
                rho = compare_images(Phi_rh_P[i], -Psi_rh_P[sub,j])
                
            corr_rh[sub] = rho
        
        corr_matrix_lh[i,j] = np.mean(corr_lh)
        corr_matrix_rh[i,j] = np.mean(corr_rh)
        
corr_matrix = np.mean((corr_matrix_lh, corr_matrix_rh), axis=0)
        
#matrix = np.triu(corr_matrix_lh, 1)

cmap = plt.get_cmap('Reds')

fig = plt.figure(figsize=(12,10), constrained_layout=False)
ax = sns.heatmap(corr_matrix, annot=False, annot_kws={'size':20},
                 cbar=False, 
                 cmap=cmap, vmin=0.0, vmax=1.)
ax.set(ylabel="")

y_labels = [f'Mode {Vn}' for Vn in range(1, 21)]
x_labels = [f'Gradient {J}' for J in range(1,21)]

ax.set_xticklabels(labels=x_labels, fontdict={'fontsize':18})
ax.set_yticklabels(labels=y_labels, fontdict={'fontsize':18})
ax.margins(0.8)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title(r'Correlation of EP gradient $\phi_f$ to EP mode $\psi_j$', fontdict={'fontsize':30})
cax = plt.axes([0.95, 0.15, 0.03, 0.7])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.ax.tick_params(labelsize=15)
cbar.set_label(label='Pearson correlation r', fontdict={'fontsize':20})
plt.show()

# =============================================================================
# #### Between-group gradient-to-gradient ####
# =============================================================================

corr_lh_phi_groups = np.zeros((Phi_lh_HC.shape[0],Phi_lh_HC.shape[0]))
corr_rh_phi_groups = np.zeros((Phi_rh_HC.shape[0],Phi_lh_HC.shape[0]))

for i in range(Phi_lh_HC.shape[0]):
    grad_lh_HC = Phi_lh_HC[i]
    grad_rh_HC = Phi_rh_HC[i]
    for j in range(Phi_lh_P.shape[0]):
        grad_lh_P = Phi_lh_P[j]
        grad_rh_P = Phi_rh_P[j]
        corr_lh = compare_images(grad_lh_HC, grad_lh_P)
        corr_rh = compare_images(grad_rh_HC, grad_rh_P)
        if corr_lh < 0:
            corr_lh = compare_images(grad_lh_HC, -grad_lh_P)
        if corr_rh < 0:
            corr_rh = compare_images(grad_rh_HC, -grad_rh_P)
        corr_lh_phi_groups[i,j] = corr_lh
        corr_rh_phi_groups[i,j] = corr_rh
        
# plot correlation
fig = plt.figure(figsize=(15,14), constrained_layout=False)
ax = sns.heatmap(corr_lh_phi_groups, annot=False, annot_kws={'size':20},
                 cbar=False, 
                 cmap=cmap, vmin=0.0, vmax=1.)
ax.set(ylabel="")
ax_labels = [f'Gradient {Vn}' for Vn in range(1, 21)]
ax.set_xticklabels(labels=ax_labels, fontdict={'fontsize':18})
labels = ax.get_yticklabels
ax.set_yticklabels(labels=ax_labels, fontdict={'fontsize':18})
ax.margins(0.8)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title(r'Correlation of HC cortical $\phi_f$ to EP cortical $\phi_f$', fontdict={'fontsize':30})
cax = plt.axes([0.95, 0.15, 0.03, 0.7])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.ax.tick_params(labelsize=15)
cbar.set_label(label='Pearson correlation coefficient', fontdict={'fontsize':30})
plt.show()
        
# =============================================================================
# #### Between group mode-to-mode ####
# =============================================================================

# TODO

corr_lh_psi_groups = np.zeros((Psi_lh_HC.shape[0],Psi_lh_HC.shape[0]))
corr_rh_psi_groups = np.zeros((Psi_rh_HC.shape[0],Psi_lh_HC.shape[0]))

for subi in range(Psi_lh_HC.shape[0]):
    mode_lh_HC = Psi_lh_HC[subi]
    mode_rh_HC = Psi_rh_HC[subi]
    for subj in range(Psi_lh_P.shape[0]):
        mode_lh_P = Psi_lh_P[subj]
        mode_rh_P = Psi_rh_P[subj]
        corr_lh = compare_images(mode_lh_HC, mode_lh_P)
        corr_rh = compare_images(mode_rh_HC, mode_rh_P)
        if corr_lh < 0:
            corr_lh = compare_images(mode_lh_HC, -mode_lh_P)
        if corr_rh < 0:
            corr_rh = compare_images(mode_rh_HC, -mode_rh_P)
        corr_lh_psi_groups[i,j] = corr_lh
        corr_rh_psi_groups[i,j] = corr_rh
        
# plot correlation
fig = plt.figure(figsize=(12,10), constrained_layout=False)
ax = sns.heatmap(corr_lh_psi_groups, annot=True, annot_kws={'size':20},
                 cbar=False, 
                 cmap=cmap, vmin=0.0, vmax=1.)
ax.set(ylabel="")

ax.set_xticklabels(labels=['Gradient 1','Gradient 2','Gradient 3'], fontdict={'fontsize':18})
labels = ax.get_yticklabels
ax.set_yticklabels(labels=['Gradient 1','Gradient 2','Gradient 3'], fontdict={'fontsize':18})
ax.margins(0.8)
plt.yticks(rotation=0)
cax = plt.axes([0.95, 0.15, 0.03, 0.7])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.ax.tick_params(labelsize=15)
cbar.set_label(label='Pearson correlation coefficient', fontdict={'fontsize':20})
plt.show()
