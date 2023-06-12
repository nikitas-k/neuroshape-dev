#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the eigenmodes of a volume

@author: James Pang and Kevin Aquino, Monash University, 2022
modified by Nikitas C. Koussis, Systems Neuroscience Group Newcastle, 2023
"""

# Import all the libraries
from lapy import Solver, TetIO
import nibabel as nib
import numpy as np
from scipy.interpolate import griddata
import os
from argparse import ArgumentParser
from neuroshape.utils.geometry import (
    get_tkrvox2ras, 
    make_tetra_file,
    normalize_vtk,
    )

def calc_eig(nifti_input_filename, output_eval_filename, output_emode_filename, num_modes, normalization_type='none', normalization_factor=1):
    """Calculate the eigenvalues and eigenmodes of the ROI volume in a nifti file.

    Parameters
    ----------
    nifti_input_filename : str
        Filename of input volume where the relevant ROI have voxel values = 1
    output_eval_filename : str  
        Filename of text file where the output eigenvalues will be stored
    output_emode_filename : str  
        Filename of text file where the output eigenmodes (in tetrahedral surface space) will be stored    
    num_modes : int
        Number of eigenmodes to be calculated
    normalization_type : str (default: 'none')
        Type of normalization
        number - normalization with respect to the total number of non-zero voxels
        volume - normalization with respect to the total volume of non-zero voxels in physical dimensions   
        constant - normalization with respect to a chosen constant
        others - no normalization
    normalization_factor : float (default: 1)
        Factor to be used in a constant normalization      

    Returns
    ------
    evals: array (num_modes x 1)
        Eigenvalues
    emodes: array (number of tetrahedral surface points x num_modes)
        Eigenmodes
    """

    # convert the ROI in the nifti file to a tetrahedral surface
    tetra_file = make_tetra_file(nifti_input_filename)

    # load tetrahedral surface (as a brainspace object)
    tetra = TetIO.import_vtk(tetra_file)

    # normalize tetrahedral surface
    tetra_norm = normalize_vtk(tetra, nifti_input_filename, normalization_type, normalization_factor)

    # calculate eigenvalues and eigenmodes
    fem = Solver(tetra_norm)
    evals, emodes = fem.eigs(k=num_modes)
    
    output_eval_file_main, output_eval_file_ext = os.path.splitext(output_eval_filename)
    output_emode_file_main, output_emode_file_ext = os.path.splitext(output_emode_filename)

    if normalization_type == 'number' or normalization_type == 'volume' or normalization_type == 'constant':
        np.savetxt(output_eval_file_main + '_norm=' + normalization_type + output_eval_file_ext, evals)
        np.savetxt(output_emode_file_main + '_norm=' + normalization_type + output_emode_file_ext, emodes)
    else:
        np.savetxt(output_eval_filename, evals)
        np.savetxt(output_emode_filename, emodes)
    
    return evals, emodes

def calc_volume_eigenmodes(nifti_input_filename, nifti_output_filename, output_eval_filename, output_emode_filename, num_modes, normalization_type='none', normalization_factor=1):
    """Main function to calculate the eigenmodes of the ROI volume in a nifti file.

    Parameters
    ----------
    nifti_input_filename : str
        Filename of input volume where the relevant ROI have voxel values = 1
    nifti_output_filename : str  
        Filename of nifti file where the output eigenmdoes (in volume space) will be stored
    output_eval_filename : str  
        Filename of text file where the output eigenvalues will be stored
    output_emode_filename : str  
        Filename of text file where the output eigenmodes (in tetrahedral surface space) will be stored    
    num_modes : int
        Number of eigenmodes to be calculated
    normalization_type : str (default: 'none')
        Type of normalization
        number - normalization with respect to the total number of non-zero voxels
        volume - normalization with respect to the total volume of non-zero voxels in physical dimensions   
        constant - normalization with respect to a chosen constant
        others - no normalization
    normalization_factor : float (default: 1)
        Factor to be used in a constant normalization
    """

    # calculate eigenvalues and eigenmodes
    evals, emodes = calc_eig(nifti_input_filename, output_eval_filename, output_emode_filename, num_modes, normalization_type, normalization_factor)


    # project eigenmodes in tetrahedral surface space into volume space

    # prepare transformation
    ROI_data = nib.load(nifti_input_filename)
    roi_data = ROI_data.get_fdata()
    inds_all = np.where(roi_data==1)
    xx = inds_all[0]
    yy = inds_all[1]
    zz = inds_all[2]

    points = np.zeros([xx.shape[0],4])
    points[:,0] = xx
    points[:,1] = yy
    points[:,2] = zz
    points[:,3] = 1

    # calculate transformation matrix
    T = get_tkrvox2ras(ROI_data.shape, ROI_data.header.get_zooms())

    # apply transformation
    points2 = np.matmul(T, np.transpose(points))

    # load tetrahedral surface
    tetra_file = nifti_input_filename + '.tetra.vtk'
    tetra = TetIO.import_vtk(tetra_file)
    points_surface = tetra.v

    # initialize nifti output array
    new_shape = np.array(roi_data.shape)
    if roi_data.ndim>3:
        new_shape[3] = num_modes
    else:
        new_shape = np.append(new_shape, num_modes)
    new_data = np.zeros(new_shape)
    
    # standardize modes
    emodes = (emodes - np.mean(emodes))/np.std(emodes)
    
    # zero baseline
    emodes = emodes - np.min(emodes)

    # perform interpolation of eigenmodes from tetrahedral surface space to volume space
    for mode in range(0, num_modes):
        interpolated_data = griddata(points_surface, emodes[:,mode], np.transpose(points2[0:3,:]), method='linear')
        for ind in range(0, len(interpolated_data)):
            new_data[xx[ind],yy[ind],zz[ind],mode] = interpolated_data[ind]

    # save to output nifti file
    img = nib.Nifti1Image(new_data, ROI_data.affine, header=ROI_data.header)
    nib.save(img, nifti_output_filename)

    # remove all created temporary auxiliary files
    geo_file = nifti_input_filename + '.geo'
    tria_file = nifti_input_filename + '.vtk'
    if os.path.exists(geo_file):
        os.remove(geo_file)
    if os.path.exists(tria_file):
        os.remove(tria_file)

def main(raw_args=None):    
    parser = ArgumentParser(epilog="volume_eigenmodes.py -- A function to calculate the eigenmodes of an ROI volume. James Pang, Monash University, 2022 <james.pang1@monash.edu>")
    parser.add_argument("nifti_input_filename", help="An input nifti with ROI voxel values=1", metavar="volume_input.nii.gz")
    parser.add_argument("nifti_output_filename", help="An output nifti where the eigenmodes in volume space will be stored", metavar="emodes.nii.gz")
    parser.add_argument("output_eval_filename", help="An output text file where the eigenvalues will be stored", metavar="evals.txt")
    parser.add_argument("output_emode_filename", help="An output text file where the eigenmods in tetrahedral surface space will be stored", metavar="emodes.txt")
    parser.add_argument("-N", dest="num_modes", default=20, help="Number of eigenmodes to be calculated, default=20", metavar="20")
    parser.add_argument("-norm", dest="normalization_type", default='none', help="Type of normalization of tetrahedral surface", metavar="none")
    parser.add_argument("-normfactor", dest="normalization_factor", default=1, help="Value of constant normalization factor of tetrahedral surface", metavar="1")

    #--------------------    Parsing the inputs from terminal:   -------------------
    args = parser.parse_args()
    nifti_input_filename    = args.nifti_input_filename
    nifti_output_filename   = args.nifti_output_filename
    output_eval_filename    = args.output_eval_filename
    output_emode_filename   = args.output_emode_filename
    num_modes               = int(args.num_modes)
    normalization_type      = args.normalization_type
    normalization_factor    = float(args.normalization_factor)
    #-------------------------------------------------------------------------------
   
    calc_volume_eigenmodes(nifti_input_filename, nifti_output_filename, output_eval_filename, output_emode_filename, num_modes, normalization_type, normalization_factor)
   

if __name__ == '__main__':
    
    # running via commandline
    main()
    

    # running within python
    # structures = ['tha']
    # hemispheres = ['lh', 'rh']
    # num_modes = 30
    # normalization_type = 'none'
    # normalization_factor = 1

    # for structure in structures:
    #     for hemisphere in hemispheres:

    #         nifti_input_filename = 'data/template_surfaces_volumes/hcp_' + structure + '-' + hemisphere + '_thr25.nii.gz'
    #         nifti_output_filename = 'data/template_eigenmodes/hcp_' structure + '-' + hemisphere + '_emode_' + str(num_modes) + '.nii.gz'
    #         output_eval_filename = 'data/template_eigenmodes/hcp' + structure + '-' + hemisphere + '_eval_' + str(num_modes) + '.txt'
    #         output_emode_filename = 'data/template_eigenmodes/hcp' + structure + '-' + hemisphere + '_emode_' + str(num_modes) + '.txt'
            
    #         calc_volume_eigenmodes(nifti_input_filename, nifti_output_filename, output_eval_filename, output_emode_filename, num_modes, normalization_type, normalization_factor):
    