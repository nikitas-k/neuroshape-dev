#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test waveshuffled surrogates
"""

from brainsmash.mapgen.sampled import Sampled
from brainsmash.mapgen.memmap import txt2memmap
from brainsmash.utils.dataio import dataio
import os
import numpy as np
import matplotlib.pyplot as plt


brain_map_file = "/Volumes/Scratch/Nik_data/brainsmash/values_naiveCC_Lmid.txt"
dist_mat_file = "/Volumes/Scratch/Nik_data/brainsmash/LeftDenseGeodesicDistmat.txt"
surrogates_folder = "/Volumes/Scratch/Nik_data/brainsmash/waveshuff_surrogates_CC_scale8/"

surr_files = [os.path.join(surrogates_folder, f) for f in os.listdir(surrogates_folder)]

x = brain_map_file
D = dist_mat_file


dist_mat_fin = dist_mat_file
output_dir = "."

#output_files = txt2memmap(dist_mat_fin, output_dir, maskfile=None, delimiter=' ')

dist_mat_mmap = 'distmat.npy'
index_mmap = 'index.npy'

sampled = Sampled(brain_map_file, dist_mat_mmap, index_mmap)

surrogate_maps = [dataio(x) for x in surr_files]
nsurr = len(surrogate_maps)

surr_var = np.empty((len(surrogate_maps), sampled.nh))

emp_var_samples = np.empty((nsurr, sampled.nh))
u0_samples = np.empty((nsurr, sampled.nh))

for i in range(nsurr):
    idx = sampled.sample()  # Randomly sample a subset of brain areas
    v = sampled.compute_variogram(sampled.x, idx)
    u = sampled.D[idx, :]
    umax = np.percentile(u, sampled.pv)
    uidx = np.where(u < umax)
    emp_var_i, u0i = sampled.smooth_variogram(
        u=u[uidx], v=v[uidx], return_h=True)
    emp_var_samples[i], u0_samples[i] = emp_var_i, u0i
    # Surrogate
    v_null = sampled.compute_variogram(surrogate_maps[i], idx)
    surr_var[i] = sampled.smooth_variogram(
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