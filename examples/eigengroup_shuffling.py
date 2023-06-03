import matplotlib.pyplot as plt
import numpy as np
from neuroshape.utils.compute_geodesic_distance import compute_geodesic_distance
from neuroshape.nulls.eigenshuff import Eigenshuff
from neuroshape.utils.dataio import dataio
from brainsmash.mapgen.memmap import txt2memmap
from lapy import TriaMesh
from lapy.ShapeDNA import compute_shapedna

cmap = plt.get_cmap('viridis')

codeFolder = "/Volumes/Scratch/functional_integration_psychosis/code/neuroshape"

# =============================================================================
# #### Surface ####
# =============================================================================

dataFolder = "/Volumes/Scratch/functional_integration_psychosis/preprocessed/HCP-EP/LBO/surfaces"
surf_file = f"{dataFolder}/1001.L.pial.32k_fs_LR.surf.gii"
coords, faces = dataio(surf_file)
tria = TriaMesh(coords, faces)
eigs = 201
eigvecs = compute_shapedna(tria, k=eigs)['Eigenvectors'][:,1:]

dist_mat_file = f'{codeFolder}/data/LeftDenseGeodesicDistmat.txt'

dist_mat_fin = dist_mat_file
output_dir = f"{codeFolder}/data/"

output_files = txt2memmap(dist_mat_fin, output_dir, maskfile=None, delimiter=' ')

dist_mat_mmap = f"{output_dir}/distmat.npy"
index_mmap = f"{output_dir}/index.npy"

# =============================================================================
# #### Compute eigenshuffled surrogates ####
# =============================================================================

generator = Eigenshuff(eigvecs, dist_mat_mmap, index_mmap, n_jobs=2)

surrogate_maps = generator(n=1000)

# plot variogram
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
ax.plot(u0, mu, color='#377eb8', label='Eigengroup-shuffled', lw=1)

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