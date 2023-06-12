#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the Laplace-Beltrami operator of a surface

     - Generates a surface if a volume nifti is given, uses Gmsh subroutines
     - Calculates the LBO and returns a number of modes
     
     NOTE: use case for both T1 brain and native surface is shown in documentation

Usual usage:
    $ data_input_filename=T1_brain.nii.gz \
    $ data_output_filename=emodes.nii.gz \
    $ output_evals_filename=evals.txt \
    $ output_emodes_filename=emodes.txt \
    $ num_modes=200 \
    $ norm_type='area' \
    $ template='fs_LR_32k' \
    $ python surface_eigenmodes.py ${data_input_filename} --eval ${output_evals_filename} --modes ${output_emodes_filename} -o ${data_output_filename} -N ${num_modes} --normalization_type=${norm_type} --resample ${template}

@author: Nikitas C. Koussis, Systems Neuroscience Group Newcastle, 2023
"""

import nibabel as nib
import numpy as np
from argparse import ArgumentParser
from nilearn import image, masking
from neuroshape.utils.normalize_data import normalize_data
from neuroshape.utils.fetch_atlas import fetch_atlas
from os.path import splitext
from lapy import TriaMesh
from lapy.ShapeDNA import compute_shapedna

image_types = [
    '.nii',
    '.nii.gz',
    '.gii',
    '.pial',
    ]

"""
Description
-----------
Wraps the Laplace-Beltrami Operator as implemented in ShapeDNA, see:
    https://en.wikipedia.org/wiki/Laplace%E2%80%93Beltrami_operator
    and:
    http://reuter.mit.edu/software/shapedna/
    
Runs the Laplace-Beltrami calculation in parallel (as implemented in
.__call__() and ._call_method()) and outputs an np.ndarray of size (n,J,N) 
of "eigenmodes" for J=200 as in Koussis et al. and where N is the number
of vertices and n is the number of files given as input to `x`.

Dependencies
------------
    'nibabel', 
    'lapy', 
    'nilearn', 
    'numpy', 
    'scipy', 
    'scikit-sparse',
    'neuromaps', 
    'nipype'
    
"""

def generate_mesh(data_input_filename):
    
    return surface


def calc_eigenmodes(surface, num_modes=2):
    coords, faces = surface
    tria = TriaMesh(coords, faces)
    ev = compute_shapedna(tria, k=num_modes)
    
    evals = ev['Eigenvalues'][1:]
    emodes = ev['Eigenvectors'][:, 1:]
    
    return evals, emodes


def surface_eigenmodes(data_input_filename, data_output_filename, output_evals_filename, output_modes_filename, num_modes=2, normalization_type='area', norm_factor=1, template=None):
    """
    Main function to calculate the connectopic Laplacian gradients of an ROI volume in NIFTI format.

    Parameters
    ----------
    data_input_filename : str
        Filename of input volume timeseries
    data_output_filename : str
        Filename of nifti file where the output gradients will be stored
    output_eval_filename : str
        Filename of text file where the output eigenvalues will be stored
    output_grad_filename : str
        Filename of text file where the output gradients will be stored
    num_modes : int, optional
        Number of non-zero modes to be calculated. The default is 2.

    """
    
    output_name, output_ext = splitext(data_input_filename)
    
    if output_ext in image_types:
        print('Loading images')
        img_input = image.load_img(data_input_filename)
        if output_ext == '.gii':
            print('Gifti image input, no need to generate surface')
            coords, faces = (img_input.darrays[0].data, img_input.darrays[1].data)
        if output_ext in ['.nii', '.nii.gz']:
            print('Nifti image input, generating surface')
            coords, faces = generate_mesh(data_input_filename, normalization_type=normalization_type, norm_factor=norm_factor)
            
    else:
        raise ValueError("Invalid input image, must be '.gii', '.nii.gz', or '.nii'")
        
        
    img_input = image.load_img(data_input_filename)
    
    evals, emodes = calc_eigenmodes(img_input, num_modes)
    
    
   
    
    
    # initialize output array
    new_shape = np.array(img_input.shape)
    new_data = np.zeros(new_shape)
    
    # put gradients into nifti array
    for mode in range(num_modes):
        data = emodes[:, mode]
        new_data[xx, yy, zz, mode] = data
        
    # save modes to output format
    print('Saving output modes file to {}'.format(data_output_filename))
    nib.save(img, data_output_filename)
    
    # write eigenvalues and gradient files
    if output_evals_filename:
        print('Saving output eigenvalues file to {}'.format(output_evals_filename))
        np.savetxt(output_evals_filename, evals)
    if output_modes_filename:
        print('Saving output gradients file to {}'.format(output_modes_filename))
        np.savetxt(output_modes_filename, emodes)
        
    if template:
        template_dict = fetch_atlas(template) #TODO

def main(raw_args=None):
    
    print('Started connectopic Laplacian, parsing command line arguments...')   
    
    parser = ArgumentParser(epilog="connectopic_laplacian.py -- A function to calculate the connectopic Laplacian of an ROI volume. Nikitas C. Koussis, Systems Neuroscience Group, 2023 <nikitas.koussis@gmail.com>")
    parser.add_argument("data_input_filename", help="An input nifti with fmri data", metavar="fmri_input.nii.gz")
    parser.add_argument("data_roi_filename", help="An input nifti with ROI voxel values=1", metavar="roi_input.nii.gz")
    parser.add_argument("mask_filename", help="An input nifti mask with voxel values=1", metavar="gm_mask.nii.gz")
    parser.add_argument("-o", dest="data_output_filename", help="An output nifti where the gradients in volume space will be stored", metavar="gradients.nii.gz")
    parser.add_argument("--eval", dest="output_evals_filename", default=None, help="An output text file where the eigenvalues will be stored", metavar="evals.txt")
    parser.add_argument("--grads", dest="output_modes_filename", default=None, help="An output text file where the gradients will be stored", metavar="grads.txt")
    parser.add_argument("-N", dest="num_gradients", default=2, help="Number of gradients to be calculated, default=2", metavar="2")
    parser.add_argument("--resample", dest="template", help="Option whether to perform smoothing and what FWHM to perform smoothing", metavar="6")
    
    #-----------------   Parsing mandatory inputs from terminal:   ----------------
    args = parser.parse_args()
    data_input_filename     = args.data_input_filename
    data_output_filename    = args.data_output_filename
    num_modes               = int(args.num_modes)
    output_evals_filename   = args.output_evals_filename
    output_modes_filename   = args.output_modes_filename
    template                = str(args.template)
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
        
    print('Inputs OK')
    
    surface_eigenmodes(data_input_filename, data_output_filename, output_evals_filename, output_modes_filename, num_modes)
    
if __name__ == '__main__':
    
    #run on command line
    main()
    
