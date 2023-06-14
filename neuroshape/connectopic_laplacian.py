#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the connectopic Laplacian of an ROI given some fmri data

Usual usage:
    $ data_input_filename=fmri_input.nii.gz \
    $ data_roi_filename=roi.nii.gz \
    $ mask_filename=mask.nii.gz \
    $ data_output_filename=gradients.nii.gz \
    $ num_gradients=20 \
    $ fwhm=6 \
    $ python connectopic_laplacian.py ${data_input_filename} ${data_roi_filename} ${mask_filename} ${data_output_filename} -N ${num_gradients} --smoothing ${fwhm} --filtering

@author: Nikitas C. Koussis, Systems Neuroscience Group Newcastle, 2023
"""

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import laplacian
from numpy.linalg import svd
from numpy.linalg import eigh
from neuroshape.utils.normalize_data import normalize_data
import os
from argparse import ArgumentParser
from nilearn import image, masking
from subprocess import Popen
from neuroshape.eta import eta_squared
from neuroshape.euler import euler_threshold
from os.path import split
import networkx as nx

os_path = dict(os.environ).get('PATH')

global matlabpath
path_list = os_path.split(sep=':')
for item in path_list:
    global matlabpath
    if 'matlab' in item:
        matlabpath = item
    if 'MATLAB' in item:
        matlabpath = item
    # else:
    #     raise RuntimeError("MATLAB is not installed or has not been sourced on the variable $PATH")

image_types = [
    '.nii',
    '.nii.gz',
]

def smooth(input_filename, fwhm):
    # load data
    img_input = image.load_img(input_filename)
    data_input = img_input.get_fdata()
    # number of volumes
    T = img_input.shape[-1]
    
    # sigma
    sigma = fwhm / np.sqrt(8 * np.log(2))
    
    # initialize smoothed image
    data_smoothed = np.zeros_like(data_input)
    
    for vol in range(T):
        data_smoothed[:,:,:,vol] = gaussian_filter(data_input[:,:,:,vol], sigma=sigma)
        
    # save smoothed nifti
    img = nib.Nifti1Image(data_smoothed, img_input.affine, header=img_input.header)
    output_filename = input_filename.replace('.nii','_smoothed.nii')
    print("Saving smoothed file to {}".format(output_filename))
    nib.save(img, output_filename)
    
    return output_filename

def wishart(input_filename, mask_filename):
    # write matlab script
    script_file = os.path.join('/tmp', 'wishart_run.m')
    folder, *_ = split(__file__)
    with open(script_file, 'w') as file:
        file.write(f"""
addpath {folder}/functions/wishart
addpath {folder}/functions

DEMDT = 1;
VN = 1;
Iterate = 2;
NDist = 1;

[~, data] = read('{input_filename}');

[~, gm_msk] = read('{mask_filename}');
ind_gm = find(gm_msk);

[folder, basename, ~] = fileparts('{input_filename}');

T = size(data,4);
x = zeros(T, length(ind_gm));

for i=1:T
    tmp = data(:,:,:,i);
    x(i,:) = tmp(ind_gm);
end

Out = icaDim(x', DEMDT, VN, Iterate, NDist);

x = Out.data';

x = detrend(x, 'constant');
x = x./repmat(std(x), T, 1);

new_data = zeros(size(gm_msk,1), size(gm_msk,2), size(gm_msk,3), T);
[xx, yy, zz] = ind2sub(size(gm_msk), ind_gm);

x = x';

for i = 1:length(ind_gm)
    new_data(xx(i), yy(i), zz(i), :) = x(i,:);
end

mat2nii(new_data, [folder, '/', basename, '_filtered.nii'], size(new_data), 32, '{mask_filename}');
                   """)
    
    # run matlab at command line
    matpath = [f'{matlabpath}']
    options = ['-nosplash', '-nodesktop', '-r']
    command = ["run('{0}'); exit;".format(script_file)]
    
    p = Popen(matpath + options + command)
    stderr, stdout = p.communicate()
        
    if stderr is not None:
        raise RuntimeWarning(stderr)
        print('Could not perform filtering, check logs')
        return input_filename
    
    os.system(f'rm -f {script_file}')  
    output_filename = input_filename.replace('.nii','_filtered.nii')
    print("Saving filtered file to {}".format(output_filename))
    
    return output_filename

def compute_similarity(img_input, img_roi, img_mask):
    data_msk = masking.apply_mask(img_input, img_mask)
    print('Normalizing input timeseries')
    data_msk = normalize_data(data_msk)
    input_img = masking.unmask(data_msk, img_mask)
    
    data_ins = masking.apply_mask(input_img, img_roi)
    
    T = data_ins.shape[0]
    
    print('Running singular value decomposition on input timeseries')
    U, S, _ = svd(data_msk, full_matrices=False)
    
    a = U.dot(np.diag(S))
    a = a[:, :-1]
    
    a = normalize_data(a)
        
    print('Computing similarity matrix by eta squared coefficient')
        
    c = data_ins.T.dot(a)
    c = c / T
    zpc = np.arctanh(c)
    zpc = zpc[:,np.all(~np.isnan(zpc), axis=0)]
    
    smat = eta_squared(zpc)
    
    return smat

def thresh(w, s):
    # # find FWHM of matrix
    # fwhm = estimate_fwhm(w)
    # # calculate resel count using FWHM
    # num_resels = 256 #resel_count(w, fwhm) DO 256 FOR NOW
    # # compute optimal threshold based on euler characteristic
    # threshold = euler_threshold(w, num_resels)
    # w_thresh = w <= threshold
    # binary = w_thresh > 0.
    # s_thresh = np.zeros_like(s)
    # s_thresh[binary] = s[binary]
    # n_edges = np.sum(np.size(s_thresh > 0.))
    # n_nodes = s_thresh.shape
    # density = 2.0*n_edges / np.prod(n_nodes)
    
    def get_components(graph):
        return nx.connected_components(graph)

    # assuming w is a 2D numpy array
    n = w.shape[0]
    
    # find upper triangle indices
    ind_upper = np.triu_indices(n, k=1)
    
    # sort w with these indices
    ind_srt = np.argsort(w[ind_upper])
    
    # initialize threshold matrix
    w_thresh = np.zeros((n, n))
    
    # define dns
    dns = np.linspace(0.001, 1, 1000)
    
    # loop through dns
    for i in range(len(dns)):
        ttl = int(np.ceil(len(ind_upper[0]) * dns[i]))
        ind_ttl = ind_srt[:ttl]

    # assign values from s to w_thresh
    w_thresh[ind_upper[0][ind_ttl], ind_upper[1][ind_ttl]] = s[ind_upper[0][ind_ttl], ind_upper[1][ind_ttl]]

    # symmetrize w_thresh
    w_thresh_sym = np.maximum(w_thresh, w_thresh.T)

    # get connected components
    graph = nx.from_numpy_array(w_thresh_sym)
    comp_sizes = list(get_components(graph))

    if len(comp_sizes) == 1:
        return w_thresh_sym

def calc_LaplacianMatrix(s, num_gradients):
    # distance mapping
    w = squareform(pdist(s))
    
    # threshold matrix
    print('Thresholding to minimum density needed for graph to remain fully connected')
    s_thresh = thresh(w, s)
    
    n_edges = len(s_thresh.flatten() > 0.)
    n_nodes = np.prod(s_thresh.shape[0], s_thresh.shape[1])
    
    density = 2*n_edges/n_nodes
    
    print('Density of similarity graph needed to remain connected: {:.2f}%'.format(density))
    
    print('Computing Laplacian')
    L = laplacian(s_thresh)
    
    return L
    
def calc_gradients(img_input, img_roi, img_mask, num_gradients=2):
    smat = compute_similarity(img_input, img_roi, img_mask)
    
    L = calc_LaplacianMatrix(smat, num_gradients)
    
    evals, egrads = eigh(L)
    
    # exclude the first eigenvalue, return only the non-zero `num_gradients` last gradients
    egrads = egrads[:, 1:num_gradients+1]
    evals = evals[1:num_gradients+1]
    
    # Z-transform eigenvectors
    egrads = normalize_data(egrads)
    
    # make positive
    egrads = egrads - np.min(egrads)
    
    
    return evals, egrads

def cortical_projection(img_input, img_roi, img_mask, egrads):
    data_msk = masking.apply_mask(img_input, img_mask)
    data_msk = normalize_data(data_msk)
    input_img = masking.unmask(data_msk, img_mask)
    
    data_ins = masking.apply_mask(input_img, img_roi)
    roi_index = [data_msk == data_ins]
    
    T = data_ins.shape[0]
    
    u, s, v = svd(data_msk, full_matrices=False)
    
    a = u * s
    a = a[:,:-1]
    
    a = normalize_data(a)
        
    c = np.matmul(data_ins.T, a)
    c = np.divide(c, T)
    
    cc = np.hstack((c, np.zeros(c.shape[0]))) * v.T
    
    index_max_cc = find_roi_index(cc, data_ins, roi_index)
    
    ind_msk = np.where(data_msk > 0.)
    eigvec_proj = np.zeros((img_mask.shape[0], img_mask.shape[1], img_mask.shape[2], egrads.shape[1]))
    
    for grad in range(egrads.shape[1]):
        eig = egrads[:, grad]
        eig_proj = eig[index_max_cc]
        xx, yy, zz = np.unravel_index(ind_msk, img_mask.shape)
        for i in data_msk.shape[0]:
            if not np.in1d(ind_msk, roi_index):
                eigvec_proj[xx[i], yy[i], zz[i], grad] = eig_proj[i]
    
    return eigvec_proj
            
    
def find_roi_index(cc, roi_data, roi_index):
    print("Finding cortex-to-roi projection by maximum correlation")
    
    max_cc = np.max(cc[roi_index])
    index_max_cc = np.zeros(max_cc.shape[0])
    for i in range(max_cc.shape[0]):
        idx = np.where(cc[:,i] == max_cc[i])
        if idx.shape[0] == 1:
            index_max_cc[i] = idx
        else:
            index_max_cc[i] = idx[0]
    
    return index_max_cc


def connectopic_laplacian(data_input_filename, data_roi_filename, mask_filename, data_output_filename, output_eval_filename, output_grad_filename, num_gradients=2, fwhm=None, filtering=False, figures=False, cortical=False):
    """
    Main function to calculate the connectopic Laplacian gradients of an ROI volume in NIFTI format.

    Parameters
    ----------
    data_input_filename : str
        Filename of input volume timeseries
    data_roi_filename : str
        Filename of input volume where the relevant ROI have voxel intensity values = 1
    mask_filename : str
        Filename of input mask of GM where the relevant voxels have intensity = 1
    data_output_filename : str
        Filename of nifti file where the output gradients will be stored
    output_eval_filename : str
        Filename of text file where the output eigenvalues will be stored
    output_grad_filename : str
        Filename of text file where the output gradients will be stored
    num_gradients : int, optional
        Number of non-zero gradients to be calculated. The default is 2.
    figures : bool, optional
        Flag whether to plot gradient figures. The default is True.

    """
    if fwhm:
        print('Smoothing with Gaussian kernel of FWHM = {}mm'.format(fwhm))
        data_input_filename = smooth(data_input_filename, fwhm)
    if filtering is True:
        print("Normalizing input data before filtering")
        img_input = image.load_img(data_input_filename)
        img_mask = image.load_img(mask_filename)
        
        # mask and normalize input data
        data_msk = masking.apply_mask(img_input, img_mask)
        data_msk = normalize_data(data_msk)
        img_input = masking.unmask(data_msk, img_mask)
        
        output_filename = data_input_filename.replace('.nii', '_normalized.nii')
        
        print('Saving normalized data to {}'.format(output_filename))
        nib.save(img_input, output_filename)
        
        print('Performing Wishart filtering using icaDim.m, see: <https://github.com/Washington-University/HCPpipelines/tree/master/global/matlab/icaDim>')
        data_input_filename = wishart(output_filename, mask_filename)
    
    print('Loading images')
    img_input = image.load_img(data_input_filename)
    img_roi = image.load_img(data_roi_filename)
    img_mask = image.load_img(mask_filename)
    
    evals, egrads = calc_gradients(img_input, img_roi, img_mask, num_gradients)
    
    # prepare nifti output array
    roi_data = img_roi.get_fdata()
    inds_all = np.where(roi_data==1)
    xx, yy, zz = inds_all
    
    # initialize nifti output array
    new_shape = np.array(roi_data.shape)
    if roi_data.ndim > 3:
        new_shape[3] = num_gradients
    else:
        new_shape = np.append(new_shape, num_gradients)
    new_data = np.zeros(new_shape)
    
    # put gradients into nifti array
    for grad in range(num_gradients):
        data = egrads[:, grad]
        new_data[xx, yy, zz, grad] = data
        
    # save gradients to nifti
    img = nib.Nifti1Image(new_data, img_roi.affine, header=img_roi.header)
    print('Saving output gradient file to {}'.format(data_output_filename))
    nib.save(img, data_output_filename)
    
    # write eigenvalues and gradient files
    if output_eval_filename:
        print('Saving output eigenvalues file to {}'.format(output_eval_filename))
        np.savetxt(output_eval_filename, evals)
    if output_grad_filename:
        print('Saving output gradients file to {}'.format(output_grad_filename))
        np.savetxt(output_grad_filename, egrads)
        
    if cortical is True:
        print('Computing cortical projection')
        eigvec_proj = cortical_projection(img_input, img_roi, img_mask, egrads)
        cort_img = nib.Nifti1Image(eigvec_proj, img_mask.affine, header=img_mask.header)
        cort_filename = data_output_filename.replace('.nii.gz','_cortical_projection.nii.gz')
        print('Saving cortical projection file to {}'.cort_filename)
        
        nib.save(cort_img, cort_filename)
        
        
    
    # TODO save figures
    


def main(raw_args=None):
    
    #fmt = "%(asctime)s,%(msecs)d %(name)-2s " "%(levelname)-2s:\n\t %(message)s"
    #datefmt = "%y%m%d-%H:%M:%S"
    
    print('Started connectopic Laplacian, parsing command line arguments...')   
    
    parser = ArgumentParser(epilog="connectopic_laplacian.py -- A function to calculate the connectopic Laplacian of an ROI volume. Nikitas C. Koussis, Systems Neuroscience Group, 2023 <nikitas.koussis@gmail.com>")
    parser.add_argument("data_input_filename", help="An input nifti with fmri data", metavar="fmri_input.nii.gz")
    parser.add_argument("data_roi_filename", help="An input nifti with ROI voxel values=1", metavar="roi_input.nii.gz")
    parser.add_argument("mask_filename", help="An input nifti mask with voxel values=1", metavar="gm_mask.nii.gz")
    parser.add_argument("-o", dest="data_output_filename", help="An output nifti where the gradients in volume space will be stored", metavar="gradients.nii.gz")
    parser.add_argument("--eval", dest="output_eval_filename", default=None, help="An output text file where the eigenvalues will be stored", metavar="evals.txt")
    parser.add_argument("--grads", dest="output_grad_filename", default=None, help="An output text file where the gradients will be stored", metavar="grads.txt")
    parser.add_argument("-N", dest="num_gradients", default=2, help="Number of gradients to be calculated, default=2", metavar="2")
    parser.add_argument("--smoothing", dest="smoothing",default=None, help="Option whether to perform smoothing and what FWHM to perform smoothing", metavar="6")
    parser.add_argument("--filter", dest="filtering", action='store_true', default=False, help="Option whether to perform Wishart filtering, INITIALIZES MATLAB SUBROUTINE. REQUIRES MATLAB 2017b OR GREATER")
    parser.add_argument("--output_figures", dest="figures", default=False, help="Option whether to output orthographic figures of gradients")
    parser.add_argument("--cortical", action='store_true', default=False, help="Option whether to output cortical projections of gradients")
    
    #-----------------   Parsing mandatory inputs from terminal:   ----------------
    args = parser.parse_args()
    data_input_filename    = args.data_input_filename
    data_roi_filename      = args.data_roi_filename
    mask_filename          = args.mask_filename
    data_output_filename   = args.data_output_filename
    num_gradients          = int(args.num_gradients)
    output_eval_filename   = args.output_eval_filename
    output_grad_filename   = args.output_grad_filename
    fwhm                   = args.smoothing
    filtering              = args.filtering
    figures                = args.figures
    cortical               = args.cortical
    #-------------------------------------------------------------------------------
    
    print("")
    print("####################### USUAL USAGE ######################")
    print("")
    print("!/bin/bash")
    print('$ data_input_filename=fmri_input.nii.gz \ ')
    print('$ data_roi_filename=roi.nii.gz \ ')
    print('$ mask_filename=mask.nii.gz \ ')
    print('$ data_output_filename=gradients.nii.gz \ ')
    print('$ num_gradients=20 \ ')
    print('$ fwhm=6 \ ')
    print('$ python connectopic_laplacian.py ${data_input_filename} ${data_roi_filename} ${mask_filename} ${data_output_filename} -N ${num_gradients} --smoothing ${fwhm} --filter --cortical')
    print("")
    print('########################### RUN ##########################')
    print("")
    print('Input fmri filename: {}'.format(data_input_filename))
    print('Input ROI filename: {}'.format(data_roi_filename))
    print('Mask filename: {}'.format(mask_filename))
    print('Output filename: {}'.format(data_output_filename))
    # if output_eval_filename:
    #     print('Output eigenvalue filename: {}'.format(output_eval_filename))
    #     if '.txt' not in output_eval_filename:
    #         *_, ext = split_filename(output_eval_filename)
    #         if ext is not None:
    #             print('Output evals file must be saved as .txt file')
    #             print('Giving evals output file .txt extension')
    #             output_eval_filename = output_eval_filename - ext + '.txt'
    #         else:
    #             print('Giving evals output file .txt extension')
    #             output_eval_filename += '.txt'
                
    # if output_grad_filename:
    #     print('Output gradients: {}'.format(output_grad_filename))
    #     if '.txt' not in output_grad_filename:
    #         *_, ext = split_filename(output_grad_filename)
    #         if ext is not None:
    #             print('Output grads file must be saved as .txt file')
    #             print('Giving grads output file .txt extension')
    #             output_grad_filename = output_grad_filename - ext + '.txt'
    #         else:
    #             print('Giving grads output file .txt extension')
    #             output_grad_filename += '.txt'
                
    print('Number of gradients to compute: {}'.format(num_gradients))
    
    if args.smoothing is not None:
        fwhm = float(args.smoothing)
        print('Performing smoothing with {}mm FWHM'.format(fwhm))
    else:
        print('Not performing smoothing')
    
    if filtering is True:
        print('Performing filtering')
        # make sure matlab is installed and on the PATH
    else:
        print('Not performing filtering')
        
    if cortical is True:
        print('Performing cortical projection')
    else:
        print('Not performing cortical projection')
        
    ### TODO FIGURES ###
    
    print('Checking inputs...')
    
    # check input file
    # if not isfile(data_input_filename):
    #     raise OSError('Could not read input file {}'.format(data_input_filename))
    # *_, ext = split_filename(data_input_filename)
    # if ext not in image_types:
    #     raise ValueError('Input file expected to be nifti with extension .nii or .nii.gz, got {}'.format(ext))
    # img_input = image.load_img(data_input_filename)
    # if img_input.ndim != 4:
    #     raise RuntimeError('Input file expected to be volume timeseries with 4 dimensions, got {} dimensions instead'.format(img_input.ndim))
    
    # if not isfile(data_roi_filename):
    #     raise OSError('Could not read ROI file {}'.format(data_roi_filename))
    # *_, ext = split_filename(data_roi_filename)
    # if ext not in image_types:
    #     raise ValueError('ROI file expected to be nifti with extension .nii or .nii.gz, got {}'.format(ext))
    # img_roi = image.load_img(data_roi_filename)
    # if np.max(img_roi.get_fdata()) != 1:
    #     raise RuntimeError('ROI image file must be a binary mask, i.e., have 1 or 0 intensity values')
    
    # if not isfile(mask_filename):
    #     raise OSError('Could not read mask file {}'.format(mask_filename))
    # *_, ext = split_filename(mask_filename)
    # if ext not in image_types:
    #     raise ValueError('Mask file expected to be nifti with extension .nii or .nii.gz, got {}'.format(ext))
    # img_mask = image.load_img(mask_filename)
    # if np.max(img_mask.get_fdata()) != 1:
    #     raise RuntimeError('Mask image file must be a binary mask, i.e., have 1 or 0 intensity values')
    
    # if (img_roi.affine==img_input.affine).all():
    #     raise ValueError("Input ROI must have the same affine transformation as the fmri input file")
    # if (img_roi.affine==img_mask.affine).all():
    #     raise ValueError("Input ROI must have the same affine transformation as the GM mask file")
    # if (img_mask.affine==img_input.affine).all():
    #     raise ValueError("Input mask must have the same affine transformation as the fmri input file")
    
    # if img_roi.shape != img_input.shape[:-1]:
    #     raise ValueError("Input ROI must have the same volume dimensions as the fmri input file")
    # if img_roi.shape != img_mask.shape:
    #     raise ValueError("Input ROI must have the same dimensions as the GM mask file")
    # if img_mask.shape != img_input.shape[:-1]:
    #     raise ValueError("Input mask must have the same volume dimensions as the fmri input file")
    
    # if image_types not in data_output_filename:
    #     *_, ext = split_filename(data_output_filename)
    #     if ext is not None:
    #         print('Data output file must be saved as .nii or .nii.gz file')
    #         print('Giving data output file .nii.gz extension')
    #         data_output_filename = data_output_filename - ext + '.nii.gz'
    #     else:
    #         print('Giving data output file .nii.gz extension')
    #         data_output_filename += '.nii.gz'
    
    # try:
    #     with open(data_output_filename, 'wb') as fh:
    #         fh.write(b'\x1f\x8b')
    #     with open(output_grad_filename, 'w') as fh:
    #         fh.write('')
    #     with open(output_eval_filename, 'w') as fh:
    #         fh.write('')
    # except:
    #     raise OSError('Could not output file into specified format, check write permissions of output folder')
    
    print('Inputs OK')
    
    connectopic_laplacian(data_input_filename, data_roi_filename, mask_filename, data_output_filename, output_eval_filename, output_grad_filename, num_gradients, fwhm, filtering, figures, cortical)
    
if __name__ == '__main__':
    
    #run on command line
    main()
    
