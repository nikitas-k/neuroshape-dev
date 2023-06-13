"""
- Split the eigenvectors of an eigengroup into half the brain and the other half
- See if the eigenvector activity of one half predicts the other half
- Do all, but expected to be medium frequency (20/30/40/50)
- Coefficients of eigenmodes to reconstruct the original data
- Are the coefficents correlated/similar at the front and back
- False positives concerns:
    - Too low frequency of the eigenmodes
- Look at the correlation of each coefficient in each eigenmode across subjects 
"""

import numpy as np
from lapy.ShapeDNA import compute_shapedna
from lapy import TriaMesh
from scipy.stats import pearsonr
from nibabel import nib
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

def calc_eigenmodes(surface_filename, n=500):
    surface = nib.load(surface_filename)
    
    # calculate the eigenmodes of a surface
    
    coords, faces = (surface.darrays[0].data, surface.darrays[1].data)
    mesh = TriaMesh(coords, faces)
    
    ev = compute_shapedna(mesh, k=n)
    
    evals = ev['Eigenvalues']
    emodes = ev['Eigenvectors']
    
    return evals, emodes

def partition_eigenmodes(emodes, partitions=2):
    # partition the eigenmodes based on the fraction of each eigenmode
    # across the cortex
    
    split_emodes = np.zeros((partitions, emodes.shape[0] // partitions, emodes.shape[1]))
    
    prev = 0
    for partition in range(partitions):
        number = emodes.shape[0] // partition
        split_emodes[partition, :] = emodes[prev:number]
        prev += number
    
    return split_emodes

def reconstruct_map(mapping, emodes):
    # reconstruct the map with split eigenmodes
    # do the weighted sum to calculate the coefficients
    # match the indices of the mapping to the vertices of the eigenmodes
    
    partitions = emodes.shape[0]
    
    coeffs = np.zeros((partitions, emodes.shape[2]))
    
    prev = 0
    for partition in range(partitions):
        number = mapping.shape[0] // partitions
        split_mapping = mapping[prev:number]
        prev += number
        
        # compute reconstruction with matrix multiplication, otherwise use regression
        try:
            coeffs[partition] = np.linalg.solve(emodes[partition].T @ emodes[partition], emodes[partition].T @ split_mapping)
        except:
            coeffs[partition] = np.linalg.lstsq(emodes[partition], split_mapping, rcond=None)[0]
    
    return coeffs

def corr_coeffs_eigengroup(coeffs, emodes):
    # correlate the coefficients within each eigengroup
    
    corr = np.zeros((coeffs.shape[0], coeffs.shape[0]))
    
    for i in range(coeffs.shape[0]):
        for j in range(i):
            corr[i,j] = pearsonr(coeffs[i], coeffs[j])
    
    # plot correlation
    cmap = plt.get_cmap('Reds')
    
    fig = plt.figure(figsize=(15,14), constrained_layout=False)
    ax = sns.heatmap(corr, annot=True, annot_kws={'size':20},
                     cbar=False, cmap=cmap, vmin=-1., vmax=1.)
    
    ax.set(ylabel="")
    ax_labels = [f'Partition {part}' for part in range(corr.shape[0])]
    ax.set_xticklabels(labels=ax_labels, fontdict={'fontsize':18})
    ax.set_yticklabels(labels=ax_labels, fontdict={'fontsize':18})
    ax.margins(0.8)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title('Correlation of coefficients in each partition of reconstruction')
    cax = plt.axes([0.95, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label='Pearson correlation coefficient', fontdict={'fontsize':30})
    plt.show()

def main(raw_args=None):
    parser = ArgumentParser(epilog="eigenmode_replication.py -- A function to partition the eigenmodes of a particular surface and calculate the correlation of each partition's coefficients to another")
    parser.add_argument("surface_input_filename", help="An input gifti surface")
    parser.add_argument("surface_map", help="A brain map with the same number of vertices as the surface")
    parser.add_argument("--num_partitions", default=2, help="Number of partitions to divide the cortex into (does this evenly)")
    parser.add_argument("--num_modes", default=200, help="Number of modes to reconstruct the original mapping with")
    
    

if __name__ == '__main__':
    # run
    main()
        
    