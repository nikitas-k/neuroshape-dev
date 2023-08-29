#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the eigenmodes of a freesurfer surface (?h.pial), outputs
in fsaverage space.

@author: Nikitas C. Koussis 2023
"""

# Import all the libraries
import numpy as np
import brainspace.mesh as mesh
import os
from argparse import ArgumentParser
from newshape.io.geometry import (
    calc_surface_eigenmodes, create_temp_surface,
    get_indices, calc_eig,
    )
from newshape.io.interfaces.freesurfer import Register
from newshape.io.interfaces.cli import freesurfer_subjects_dir
from newshape.io.iosupport import read_fs, _read_annot, write_gifti, split_filename
from newshape.mesh.mesh_operations import apply_mask
from newshape.shape import Shape

def calc_fs_eigenmodes(surface_input_filename, subject, subjects_dir, output_eval_filename, 
                  output_emode_filename, save_cut=True, save_mask=True, num_modes=20):
    """
    Calculate the eigenmodes of a freesurfer hemisphere file, removing the 
    medial wall. The surface is saved in the same output directory as the 
    eigenmodes and eigenvalues. If --save_cut is passed, then surface with 
    cuts made is saved as a .vtk object in the same directory.

    Parameters
    ----------
    surface_input_filename : str
        Path to freesurfer triangle file
    subject : str
        Subject name in `subjects_dir`
    subjects_dir : str
        Path to freesurfer subjects directory
    output_eval_filename : str
        Path for eigenvalues output
    output_emode_filename : str
        Path for eigenmodes output without medial wall
    save_cut : bool, optional
        Whether to save the cut surface. The default is True.
    save_mask : bool, optional
        Whether to save the mask. The default is True.
    num_modes : int, optional
        Number of eigenmodes to compute. The default is 20.

    Notes
    -----
    Eigenmodes are resampled across the original surface made before the removal
    of the medial wall.

    """
    
    is_ascii = False
    
    # check output files
    outevdir, evfile, ext = split_filename(output_eval_filename)
    if not outevdir:
        outevdir = os.getcwd()
    if ext != '.txt':
        raise ValueError('Output eigenvalue file must have .txt extension')
    
    outemdir, emfile, ext = split_filename(output_emode_filename)
    if not outemdir:
        outemdir = os.getcwd()
    if ext != '.txt':
        raise ValueError('Output eigenmodes file must have .txt extension')
    
    # get hemisphere and surface
    outdir, surffile, ext = split_filename(surface_input_filename)
        
    splitsurf = surface_input_filename.split(".", 1)
    hemi = splitsurf[0]
    surf = splitsurf[1]
    
    # read in surface
    if ext not in ['.pial', '.white', '.midthickness', 
                   '.inflated', '.sphere', '.asc']:
        raise ValueError('Input file must be freesurfer geometry file, e.g., ?h.pial')
    
    if ext == 'asc':
        is_ascii=True
    
    if hemi == 'lh':
        outhemi = 'L'
    else:
        outhemi = 'R'
    
    surface_orig = read_fs(surface_input_filename)
    surface_orig = Shape(surface_orig)
    
    # write out as outemdir/subject.(LR).native.(surf).gii to match 
    # workbench naming structure
    surface_orig.write_surface(filename=os.path.join(outemdir, subject + '.' + outhemi + '.native.' + surf + '.gii'))
    
    # find indices in annot file that match medial wall
    annot_file = os.path.join(subjects_dir, subject, 'label', hemi + '.aparc.a2009s.annot')
    labels, *_ = _read_annot(annot_file)
    mask = np.argwhere(labels==-1)
    
    if save_mask:
        np.savetxt(os.path.join(outemdir, subject + '_' + hemi + '_mask.txt'))
    
    # make cut
    surface_cut = apply_mask(surface_orig, mask)
    
    if save_cut:
        surface_cut.write_surface(filename=os.path.join(outemdir, subject + '.' + outhemi + '.native.' + surf + '_cut.gii'))
    
    # compute eigenmodes
    evals, emodes = surface_cut.compute_eigenmodes(num_modes=num_modes)
    
    # get indices of vertices of surface_orig that match surface_cut
    indices = get_indices(surface_orig, surface_cut)
    
    # reshape emodes to match original surface
    emodes_reshaped = np.zeros([surface_orig.num_verts, np.shape(emodes)[1]])
    for mode in range(np.shape(emodes)[1]):
        emodes_reshaped[indices, mode] = np.expand_dims(emodes[:, mode], axis=1)
    
    # save evals and emodes
    np.savetxt(output_eval_filename, evals)
    np.savetxt(output_emode_filename, emodes_reshaped)
    
    # return 0 to quit out
    print('Processing complete, eigenmodes output to {}'.format(output_emode_filename))
    return


def main(raw_args=None):    
    parser = ArgumentParser(epilog="fs_eigenmodes.py -- A function to calculate the eigenmodes of a cortical surface. James Pang, Monash University, 2022 <james.pang1@monash.edu>")
    parser.add_argument("surface_input_filename", help="An input surface in freesurfer format", metavar="?h.pial")
    parser.add_argument("subject", help="Input subject in $SUBJECTS_DIR")
    parser.add_argument("fs_subjects_dir", help="Freesurfer subjects directory, if not given then is guessed from environment variable $SUBJECTS_DIR")
    parser.add_argument("output_eval_filename", help="An output text file where the eigenvalues will be stored", metavar="evals.txt")
    parser.add_argument("output_emode_filename", help="An output text file where the eigenmodes will be stored", metavar="emodes.txt")
    parser.add_argument("--save_cut", dest="save_cut", action="store_true", help="Logical value to decide whether to write the masked version of the input surface")
    parser.add_argument("--save_mask", dest="save_mask", action="store_true", help="Logical to decide whether or not to save mask file")
    parser.add_argument("-N", dest="num_modes", default=20, help="Number of eigenmodes to be calculated, default=20", metavar="20")
    
    #--------------------    Parsing the inputs from terminal:   -------------------
    args = parser.parse_args()
    surface_input_filename   = args.surface_input_filename
    subject                  = args.subject
    subjects_dir             = args.fs_subjects_dir
    output_eval_filename     = args.output_eval_filename
    output_emode_filename    = args.output_emode_filename
    save_cut                 = args.save_cut
    save_mask                = args.save_mask
    num_modes                = int(args.num_modes)
    #-------------------------------------------------------------------------------
    
    if args.fs_subjects_dir is None:
        subjects_dir = freesurfer_subjects_dir()
        if not subjects_dir:
            raise RuntimeError('Cannot find subjects directory, check proper sourcing')
    
    calc_fs_eigenmodes(surface_input_filename, subject, subjects_dir, 
                  output_eval_filename, output_emode_filename, 
                  save_cut, save_mask, num_modes)
    
   
if __name__ == '__main__':
    
    # running via commandline
    main()
    

    # # running within python
    # surface_interest = 'fsLR_32k'
    # structure = 'midthickness'
    # hemispheres = ['lh', 'rh']
    # num_modes = 200
    # save_cut = 0
    
    # for hemisphere in hemispheres:
    #     print('Processing ' + hemisphere)

    #     surface_input_filename = 'data/template_surfaces_volumes/' + surface_interest + '_' + structure + '-' + hemisphere + '.vtk'
    #     mask_input_filename = 'data/template_surfaces_volumes/' + surface_interest + '_cortex-' + hemisphere + '_mask.txt'
        
    #     # with cortex mask (remove medial wall)
    #     # this is the advisable way
    #     output_eval_filename = 'data/template_eigenmodes/' + surface_interest + '_' + structure + '-' + hemisphere + '_eval_' + str(num_modes) + '.txt'
    #     output_emode_filename = 'data/template_eigenmodes/' + surface_interest + '_' + structure + '-' + hemisphere + '_emode_' + str(num_modes) + '.txt'

    #     calc_surface_eigenmodes(surface_input_filename, mask_input_filename, output_eval_filename, output_emode_filename, save_cut, num_modes)
        
    #     # without cortex mask
    #     output_eval_filename = 'data/template_eigenmodes/' + 'nomask_' + surface_interest + '_' + structure + '-' + hemisphere + '_eval_' + str(num_modes) + '.txt'
    #     output_emode_filename = 'data/template_eigenmodes/' + 'nomask_' + surface_interest + '_' + structure + '-' + hemisphere + '_emode_' + str(num_modes) + '.txt'

    #     calc_surface_eigenmodes_nomask(surface_input_filename, output_eval_filename, output_emode_filename, num_modes)
            
 
