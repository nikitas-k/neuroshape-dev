#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test generatorled surrogates
"""

#from brainsmash.mapgen.generator import Sampled
#from brainsmash.mapgen.memmap import txt2memmap
#from brainsmash.utils.dataio import dataio
#import os
import numpy as np
import matplotlib.pyplot as plt
from brainsmash.mapgen.waveshuff import Waveshuff
from brainsmash.utils.concavehull import ConcaveHull
from brainsmash.utils.load_mesh_vertices import load_mesh_vertices


brain_map_file = "/Volumes/Scratch/Nik_data/brainsmash/values_naiveCC_Lmid.txt" #/Volumes/Scratch/Nik_data/MNINonLinear/fsaverage_LR32k/1001_01_MR.L.thickness.32k_fs_LR.shape.gii"
dist_mat_file = "/Volumes/Scratch/Nik_data/brainsmash/LeftDenseGeodesicDistmat.txt"
#surrogates_folder = "/Volumes/Scratch/Nik_data/brainsmash/generator_surrogates_CC_scale14/"

#surr_files = [os.path.join(surrogates_folder, f) for f in os.listdir(surrogates_folder)]

x = brain_map_file
D = dist_mat_file


dist_mat_fin = dist_mat_file
output_dir = "."

#output_files = txt2memmap(dist_mat_fin, output_dir, maskfile=None, delimiter=' ')

dist_mat_mmap = 'distmat.npy'
index_mmap = 'index.npy'

vertices = load_mesh_vertices("/Volumes/Scratch/Nik_data/MNINonLinear/fsaverage_LR32k/1001_01_MR.L.flat.32k_fs_LR.surf.gii")

ch = ConcaveHull()
ch.loadpoints(vertices[:, :2])
ch.calculatehull(tol=30)

generator = Waveshuff(brain_map_file, dist_mat_mmap, index_mmap, 
                      n_jobs=20, wv='db4', ch=ch, scales=np.arange(8))

surrogate_maps = generator(n=1000)
nsurr = len(surrogate_maps)

surr_var = np.empty((len(surrogate_maps), generator.nh))

emp_var_samples = np.empty((nsurr, generator.nh))
u0_samples = np.empty((nsurr, generator.nh))

for i in range(nsurr):
    idx = generator.sample()  # Randomly sample a subset of brain areas
    v = generator.compute_variogram(generator.x, idx)
    u = generator.D[idx, :]
    umax = np.percentile(u, generator.pv)
    uidx = np.where(u < umax)
    emp_var_i, u0i = generator.smooth_variogram(
        u=u[uidx], v=v[uidx], return_h=True)
    emp_var_samples[i], u0_samples[i] = emp_var_i, u0i
    # Surrogate
    v_null = generator.compute_variogram(surrogate_maps[i], idx)
    surr_var[i] = generator.smooth_variogram(
        u=u[uidx], v=v_null[uidx], return_h=False)

u0 = u0_samples.mean(axis=0)
emp_var = emp_var_samples.mean(axis=0)

# Plot target variogram
fig = plt.figure(figsize=(5, 5))
ax = fig.add_axes([0.12, 0.15, 0.8, 0.77])
ax.scatter(u0, emp_var, s=20, facecolor='none', edgecolor='k',
           marker='o', lw=1, label='Empirical')

# Plot surrogate maps' variograms
mu = surr_var.mean(axis=0)
sigma = surr_var.std(axis=0)
ax.fill_between(u0, mu-sigma, mu+sigma, facecolor='#377eb8',
                edgecolor='none', alpha=0.3)
ax.plot(u0, mu, color='#377eb8', label='Wavestrapped', lw=1)

# Make plot nice
leg = ax.legend(loc=0)
leg.get_frame().set_linewidth(0.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.setp(ax.get_yticklabels(), visible=False)
# plt.setp(ax.get_yticklines(), visible=False)
# plt.setp(ax.get_xticklabels(), visible=False)
# plt.setp(ax.get_xticklines(), visible=False)
ax.set_xlabel("Spatial separation\ndistance")
ax.set_ylabel("Variance")
plt.show()