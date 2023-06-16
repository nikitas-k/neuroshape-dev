#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the Laplace-Beltrami operator of a surface

     - Generates a surface if a volume nifti is given, uses Gmsh subroutines
     - Calculates the LBO and returns a number of modes
     
     NOTE: use case for both T1 brain and native surface is shown in documentation

@author: Nikitas C. Koussis, Systems Neuroscience Group Newcastle, 2023, based
on fs_shapeDNA.py by Martin Reuter.
"""

import nibabel as nib
import numpy as np
from argparse import ArgumentParser
from neuroshape.utils.fetch_atlas import fetch_atlas
from os.path import splitext
from lapy import TriaMesh, TriaIO, Solver
from neuroshape.utils.geometry import (
    normalize_vtk,
    make_tetra_file,
    get_tkrvox2ras,
    mri_mc,    
    read_geometry,
    read_label,
    )
import optparse
import sys
import os
import tempfile
import warnings
import errno

warnings.filterwarnings('ignore', '.*negative int.*')
os.environ['OMP_NUM_THREADS'] = '1' # setenv OMP_NUM_THREADS 1

image_types = [
    '.nii',
    '.nii.gz',
    '.gii',
    '.pial',
    ]

norm_types = [
    'area',
    'constant',
    'number',
    'volume',   
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

def my_print(message):
    """
    print message, then flush stdout
    """
    print(message)
    sys.stdout.flush()

HELPTEXT = """

SUMMARY

Computes surface (and volume) models of FreeSurfer's subcortical and
cortical structures and computes a spectral shape descriptor
(ShapeDNA [1]).

Input can be one of the following:
--asegid  : an aseg label id to construct a surface of that ROI
--surf    : a surface, e.g. lh.white
--label   : a label file to cut out a surface patch (also pass --surf)
--aparcid : an aparc id to cut out that cortical ROI (also pass --surf)

If "meshfix" and "gmsh" are in the path, and shapeDNA-tetra is available
in the $SHAPEDNA_HOME directory, it can also create and process
tetrahedral meshes from closed surfaces, such as lh.white ( --dotet ).

REFERENCES
==========

[1] M. Reuter, F.-E. Wolter and N. Peinecke.
Laplace-Beltrami spectra as "Shape-DNA" of surfaces and solids.
Computer-Aided Design 38 (4), pp.342-366, 2006.
http://dx.doi.org/10.1016/j.cad.2005.10.011

"""

TMPTXT="""
REQUIRED ARGUMENTS

--sid <name>      Subject ID

One of the following:

--asegid <int>    Segmentation ID of structure in aseg.mgz (e.g. 17 is
                  Left-Hippocampus), for ID's check <sid>/stats/aseg.stats
                  or $FREESURFER_HOME/FreeSurferColorLUT.txt.

--surf <name>     lh.pial, rh.pial, lh.white, rh.white etc. to select a 
                  surface from the <sid>/surfs directory.


OPTIONAL ARGUMENTS

--sdir <name>     Subjects directory (or set via environment $SUBJECTS_DIR)

--outdir <name>   Output directory (default: <sdir>/<sid>/brainprint/ )

--outsurf <name>  Name for surface output (with --asegid)
                  (default: aseg.<asegid>.surf )

--outev <name>    Name for eigenvalue output
                  (default: <(out)surf>.<shapedna_parametrs>.ev )

ShapeDNA parameters (see shapeDNA-tria --help for details):

--num <int>       Number of eigenvalues/vectors to compute (default: 50)

--degree <int>    FEM degree (default 1)

--bcond <int>     Boundary condition (0=Dirichlet, 1=Neumann default)

--evec            Additionally compute eigenvectors

--ignorelq        Ignore low quality in input mesh

--sparam "<str>"  Additional parameters for shapeDNA-tria
"""

def split_callback(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))
  
def options_parse():
    """
    Command Line Options Parser:
    initiate the option parser and return the parsed object
    """
    parser = optparse.OptionParser(version='$Id: eigenmodes, N Koussis $', usage=HELPTEXT)
    
    # help text
    h_sid       = '(REQUIRED) subject ID (FS processed directory inside the subjects directory)'
    h_sdir      = 'FS subjects directory (or set environment $SUBJECTS_DIR)'
    
    h_asegid    = 'segmentation ID of structure in aseg.mgz (e.g. 17 is Left-Hippocampus), for ID\'s check <sid>/stats/aseg.stats or $FREESURFER_HOME/FreeSurferColorLUT.txt'
    h_surf      = 'surface name, e.g. lh.pial, rh.pial, lh.white, rh.white etc. to select a surface from the <sid>/surfs directory'
    h_aparcid   = 'segmentation ID of structure in aparc (e.g. 24 is precentral), requires --surf (including the hemi prefix), for ID\'s check $FREESURFER_HOME/average/colortable_desikan_killiany.txt'
    h_label     = 'full path to label file, to create surface patch, requires --surf (including the hemi prefix)'
    h_source    = 'specify source subject with --label (to map label e.g. from fsaverage space)'

    h_dotet     = 'construct a tetrahedra volume mesh and compute spectra of solid'
    h_fixiter   = 'iterations of meshfix (default=4), only with --dotet'
    
    h_outdir    = 'Output directory (default: <sdir>/<sid>/brainprint/ )'
    h_outsurf   = 'Full path for surface output in VTK format (with --asegid default: <outdir>/aseg.<asegid>.vtk )'
    h_outtet    = 'Full path for tet output (with --dotet) (default: <outdir>/<(out)surf>.msh )'
    h_outev     = 'Full path for eigenvalue output (default: <outdir>/<(out)surf or outtet>.ev )'
    
    h_num       = 'number of eigenvalues/vectors to compute (default: 50)'
    h_degree    = 'degree for FEM computation (1=linear default, 3=cubic)'
    h_bcond     = 'boundary condition (1=Neumann default, 0=Dirichlet )'
    h_evec      = 'bool to switch on eigenvector computation'
    h_ignorelq  = 'ignore low quality in input mesh'
    h_refmin    = 'mesh refinement so that DOF is at least <int>'
    h_tsmooth   = 'tangential smoothing iterations (after refinement)'
    h_gsmooth   = 'geometry smoothing iterations (after refinement)'
    h_param2d   = 'additional parameters for shapeDNA-tria'
   
    # Add options 

    # Sepcify inputs
    parser.add_option('--sdir', dest='sdir', help=h_sdir)

    group = optparse.OptionGroup(parser, "Required Options", "Specify --sid and select one of the other options")
    group.add_option('--sid',     dest='sid',      help=h_sid)
    group.add_option('--asegid',  dest='asegid' ,  help=h_asegid,  type='string', action='callback', callback=split_callback)
    group.add_option('--aparcid', dest='aparcid' , help=h_aparcid, type='string', action='callback', callback=split_callback)
    group.add_option('--surf' ,   dest='surf' ,    help=h_surf)
    group.add_option('--label' ,  dest='label' ,   help=h_label)
    parser.add_option_group(group)


    group = optparse.OptionGroup(parser, "Additional Flags", )
    group.add_option('--source' , dest='source',   help=h_source)
    group.add_option('--dotet' ,  dest='dotet',    help=h_dotet,   action='store_true', default=False)
    group.add_option('--fixiter', dest='fixiter',  help=h_fixiter, default=4, type='int')
    parser.add_option_group(group)

    #output switches
    group = optparse.OptionGroup(parser, "Output Parameters" )
    group.add_option('--outdir',  dest='outdir',   help=h_outdir)
    group.add_option('--outsurf', dest='outsurf',  help=h_outsurf)
    group.add_option('--outtet',  dest='outtet',   help=h_outtet)
    group.add_option('--outev',   dest='outev',    help=h_outev)
    parser.add_option_group(group)

    #shapedna switches
    group = optparse.OptionGroup(parser, "ShapeDNA Parameters","See shapeDNA-tria --help for details")
    group.add_option('--num' ,     dest='num',      help=h_num,      default=50, type='int')
    group.add_option('--degree' ,  dest='degree',   help=h_degree,   default=1,  type='int')
    group.add_option('--bcond' ,   dest='bcond',    help=h_bcond,    default=1,  type='int')
    group.add_option('--evec' ,    dest='evec',     help=h_evec,     default=False, action='store_true' )
    group.add_option('--ignorelq', dest='ignorelq', help=h_ignorelq, default=False, action='store_true')
    group.add_option('--refmin',   dest='refmin',   help=h_refmin,   default=0,  type='int')
    group.add_option('--tsmooth',  dest='tsmooth',  help=h_tsmooth,  default=0,  type='int')   
    group.add_option('--gsmooth',  dest='gsmooth',  help=h_gsmooth,  default=0,  type='int')   
    group.add_option('--param2d',  dest='param2d',  help=h_param2d)
    parser.add_option_group(group)
    
                      
    (options, args) = parser.parse_args()
    
    if options.sdir is None:
        options.sdir = os.getenv('SUBJECTS_DIR')
        
    if options.sdir is None:
        parser.print_help()
        my_print('\nERROR: specify subjects directory via --sdir or $SUBJECTS_DIR\n')
        sys.exit(1)
        
    if options.sid is None:
        parser.print_help()
        my_print('\nERROR: Specify --sid\n')
        sys.exit(1)
        
    subjdir = os.path.join(options.sdir,options.sid)
    if not os.path.exists(subjdir):
        my_print('ERROR: cannot find sid in subjects directory\n')
        sys.exit(1)
        
    if options.label is not None and options.surf is None:
        parser.print_help()
        my_print('\nERROR: Specify --surf with --label\n')
        sys.exit(1)  
    if options.aparcid is not None and options.surf is None:
        parser.print_help()
        my_print('\nERROR: Specify --surf with --aparc\n')
        sys.exit(1)  
    # input needs to be either a surf or aseg label(s)
    if options.asegid is None and options.surf is None:
        parser.print_help()
        my_print('\nERROR: Specify either --asegid or --surf\n')
        sys.exit(1)
    # and it cannot be both
    if options.asegid is not None and options.surf is not None:
        parser.print_help()
        my_print('\nERROR: Specify either --asegid or --surf (not both)\n')
        sys.exit(1)  
    
    # set default output dir (maybe this should be ./ ??)
    if options.outdir is None:
        options.outdir = os.path.join(subjdir,'neuroshape')
    try:
        os.mkdir(options.outdir)
    except OSError as e:
        if e.errno != os.errno.EEXIST:
            raise e
        pass

    # check if we have write access to output dir
    try:
        testfile = tempfile.TemporaryFile(dir = options.outdir)
        testfile.close()
    except OSError as e:
        if e.errno != errno.EACCES:  # 13
            e.filename = options.outdir
            raise
        my_print('\nERROR: '+options.outdir+' not writeable (check access)!\n')
        sys.exit(1)
    
    # initialize outsurf
    if options.outsurf is None:
        # for aseg stuff, we need to create a surface (and we'll keep it around)
        if options.asegid is not None:
            astring  = '_'.join(options.asegid)
            #surfname = 'aseg.'+astring+'.surf'
            surfname = 'aseg.'+astring+'.vtk'
            options.outsurf = os.path.join(options.outdir,surfname)    
        elif options.label is not None:
            surfname = os.path.basename(options.surf)+'.'+os.path.basename(options.label)+'.vtk'
            options.outsurf = os.path.join(options.outdir,surfname)
        elif options.aparcid is not None:
            astring  = '_'.join(options.aparcid)
            surfname = os.path.basename(options.surf)+'.aparc.'+astring+'.vtk'
            options.outsurf = os.path.join(options.outdir,surfname)          
        else:
            # for surfaces, a refined/smoothed version could be written
            surfname = os.path.basename(options.surf)+'.vtk'
            options.outsurf = os.path.join(options.outdir,surfname)
    else:
        # make sure it is vtk ending
        if (os.path.splitext(options.outsurf)[1]).upper() != '.VTK':
            my_print('ERROR: outsurf needs vtk extension (VTK format)')
            sys.exit(1)
    
    # for 3d processing, initialize outtet:
    if options.dotet and options.outtet is None:
        surfname = os.path.basename(options.outsurf)
        options.outtet = os.path.join(options.outdir,surfname+'.msh')    
    
    # set source to sid if empty
    if options.source is None:
        options.source = options.sid
        
    # if label does not exist, search in subject label dir
    if options.label is not None and not os.path.isfile(options.label):
        ltemp = os.path.join(options.sdir,options.source,'label',options.label)
        if os.path.isfile(ltemp):
            options.label = ltemp
        else:
            parser.print_help()
            my_print('\nERROR: Specified --label not found\n')
            sys.exit(1)  
                
    # initialize outev 
    if options.outev is None:
        if options.dotet:
            options.outev = options.outtet+'.txt'
        else:
            options.outev = options.outsurf+'.txt'
        
    return options

# Gets global path to surface input (if it is a FS surf)
def get_path_surf(sdir,sid,surf):
    return os.path.join(sdir,sid,'surf',surf)

# creates tria surface from label (and specified surface)
# maps the label first if source is different from sid
def make_surf(sdir,sid,surf,outdir):
    # get hemi from surf
    splitsurf = surf.split(".",1)
    hemi = splitsurf[0]
    surf = splitsurf[1]
    # map label if necessary
    coords, faces = read_geometry(get_path_surf(sdir, sid, surf))
    
    outsurf = os.path.join(outdir, hemi, surf, '.vtk')
    
    # save tria mesh to outsurf
    tria = TriaMesh(coords, faces)
    
    TriaIO.export_vtk(tria, outsurf)
    
    # return surf name
    return outsurf

# creates tria surface from label (and specified surface)
# maps the label first if source is different from sid
# def get_label_surf(sdir,sid,label,surf,source,outsurf):
#     subjdir  = os.path.join(sdir,sid)
#     outdir   = os.path.dirname( outsurf )
#     stlsurf  = os.path.join(outdir,'label'+str(uuid.uuid4())+'.stl')

#     # get hemi from surf
#     splitsurf = surf.split(".",1)
#     hemi = splitsurf[0]
#     surf = splitsurf[1]
#     # map label if necessary
#     mappedlabel = label
#     if (source != sid):
#         mappedlabel = os.path.join(outdir,os.path.basename(label)+'.'+str(uuid.uuid4())+'.label')
#         cmd = 'mri_label2label --sd '+sdir+' --srclabel '+label+' --srcsubject '+ source+' --trgsubject '+sid+' --trglabel '+mappedlabel+' --hemi '+hemi+' --regmethod surface'
#         run_cmd(cmd,'mri_label2label failed?')     
#     # make surface path (make sure output is stl, this currently needs fsdev, will probably be in 6.0)
#     cmd = 'label2patch -writesurf -sdir '+sdir+' -surf '+ surf + ' ' +sid+' '+hemi+' '+mappedlabel+' '+stlsurf
#     run_cmd(cmd,'label2patch failed?')     
#     cmd = 'mris_convert '+stlsurf+' '+outsurf
#     run_cmd(cmd,'mris_convert failed.')
    
#     # cleanup map label if necessary
#     if (source != sid):
#         cmd ='rm '+mappedlabel
#         run_cmd(cmd,'rm mapped label '+mappedlabel+' failed.')     
#     cmd = 'rm '+stlsurf
#     run_cmd(cmd,'rm stlsurf failed.')
#     # return surf name
#     return outsurf

# Creates a surface from the aseg and label info
# and writes it to the outdir
# def get_aseg_surf(sdir,sid,asegid,outsurf):
#     astring2 = ' '.join(asegid)
#     subjdir  = os.path.join(sdir,sid)
#     aseg     = os.path.join(subjdir,'mri','aseg.mgz')
#     norm     = os.path.join(subjdir,'mri','norm.mgz')  
#     outdir   = os.path.dirname( outsurf )    
#     tmpname  = 'aseg.'+str(uuid.uuid4())
#     segf     = os.path.join(outdir,tmpname+'.mgz')
#     segsurf  = os.path.join(outdir,tmpname+'.surf')
#     # binarize on selected labels (creates temp segf)
#     ptinput = aseg
#     ptlabel = str(asegid[0])
#     #if len(asegid) > 1:
#     # always binarize first, otherwise pretess may scale aseg if labels are larger than 255 (e.g. aseg+aparc, bug in mri_pretess?)
#     cmd ='mri_binarize --i '+aseg+' --match '+astring2+' --o '+segf
#     run_cmd(cmd,'mri_binarize failed.') 
#     ptinput = segf
#     ptlabel = '1'
#     # if norm exist, fix label (pretess)
#     if os.path.isfile(norm):
#         cmd ='mri_pretess '+ptinput+' '+ptlabel+' '+norm+' '+segf
#         run_cmd(cmd,'mri_pretess failed.') 
#     else:
#         if not os.path.isfile(segf):
#             # cp segf if not exist yet
#             # (it exists already if we combined labels above)
#             cmd = 'cp '+ptinput+' '+segf
#             run_cmd(cmd,'cp segmentation file failed.') 
#     # runs marching cube to extract surface
#     cmd ='mri_mc '+segf+' '+ptlabel+' '+segsurf
#     run_cmd(cmd,'mri_mc failed?') 
#     # convert to stl
#     cmd ='mris_convert '+segsurf+' '+outsurf
#     run_cmd(cmd,'mris_convert failed.')
#     # cleanup temp files
#     cmd ='rm '+segf
#     run_cmd(cmd,'rm temp segf failed.') 
#     cmd ='rm '+segsurf
#     run_cmd(cmd,'rm temp segsurf failed.') 
#     # return surf name
#     return outsurf

# # Creates a surface from the aparc and label number
# # and writes it to the outdir
# def get_aparc_surf(sdir,sid,surf,aparcid,outsurf):
#     astring2 = ' '.join(aparcid)
#     subjdir  = os.path.join(sdir,sid)
#     outdir   = os.path.dirname( outsurf )    
#     rndname = str(uuid.uuid4()) 
#     # get hemi from surf
#     hemi = surf.split(".",1)[0]
#     # convert annotation id to label:
#     alllabels = ''
#     for aid in aparcid:
#         # create label of this id
#         outlabelpre = os.path.join(outdir,hemi+'.aparc.'+rndname)
#         cmd = 'mri_annotation2label --sd '+sdir+' --subject '+sid+' --hemi '+hemi+' --label '+str(aid)+' --labelbase '+outlabelpre 
#         run_cmd(cmd,'mri_annotation2label failed?') 
#         alllabels = alllabels+'-i '+outlabelpre+"-%03d.label" % int(aid)+' ' 
#     # merge labels (if more than 1)
#     mergedlabel = outlabelpre+"-%03d.label" % int(aid)
#     if len(aparcid) > 1:
#         mergedlabel = os.path.join(outdir,hemi+'.aparc.all.'+rndname+'.label')
#         cmd = 'mri_mergelabels '+alllabels+' -o '+mergedlabel
#         run_cmd(cmd,'mri_mergelabels failed?') 
#     # make to surface (call subfunction above)
#     get_label_surf(sdir,sid,mergedlabel,surf,sid,outsurf)
#     # cleanup
#     if len(aparcid) > 1:
#         cmd ='rm '+mergedlabel
#         run_cmd(cmd,'rm '+mergedlabel+' failed?')
#     for aid in aparcid:
#         outlabelpre = os.path.join(outdir,hemi+'.aparc.'+rndname+"-%03d.label" % int(aid))
#         cmd ='rm '+outlabelpre
#         run_cmd(cmd,'rm '+outlabelpre+' failed?')
#     # return surf name
#     return outsurf

# def get_tetmesh(surf,outtet,fixiter):
#     surfbase  = os.path.basename(surf)
#     outdir    = os.path.dirname( outtet )    
#     surftemp_stl = os.path.join(outdir,surfbase+'.temp.stl')
    
#     # massage surface mesh (rm handles, 60000 vertices, uniform)
#     cmd='mris_convert '+surf+' '+surftemp_stl
#     run_cmd(cmd,'mris_convert (to STLs) failed')
# #    cmd='meshfix '+surftemp_stl+' -a 2.0 --remove-handles -q --stl -o '+surftemp_stl
# #    run_cmd(cmd,'meshfix (remove-handles) failed, is it in your path?') 
#     cmd='meshfix '+surftemp_stl+' -a 2.0 -u 5 --vertices 60000 -q --stl -o '+surftemp_stl
#     run_cmd(cmd,'meshfix (downsample) failed?') 
#     for num in range(0,fixiter):
#         cmd='meshfix '+surftemp_stl+' -a 2.0 -u 1 -q --stl -o '+surftemp_stl
#         run_cmd(cmd,'meshfix failed ('+str(num)+'a)?')      
#         cmd='meshfix '+surftemp_stl+' -a 2.0 -q --stl -o '+surftemp_stl
#         run_cmd(cmd,'meshfix failed ('+str(num)+'b)?') 
        
# # replacing meshfix never worked:
# #    sdnahome = os.getenv('SHAPEDNA_HOME')
# #    triaio = os.path.join(sdnahome,"triaIO")  
# #    cmd=triaio+' --infile '+surf+' --outfile '+surftemp_stl' --contractedges --remeshbk 1'
# #    run_cmd(cmd,'triaIO remeshing failed?')
      
#     # write gmsh geofile
#     geofile  = os.path.join(outdir,'gmsh.'+str(uuid.uuid4())+'.geo')
#     g = open(geofile, 'w')
#     g.write("Mesh.Algorithm3D=4;\n")
#     g.write("Mesh.Optimize=1;\n")
#     g.write("Mesh.OptimizeNetgen=1;\n")
#     g.write("Merge \"" + surfbase+'.temp.stl' + "\";\n")
#     g.write("Surface Loop(1) = {1};\n")
#     g.write("Volume(1) = {1};\n")
#     g.write("Physical Volume(1) = {1};\n")
#     g.close()
#     # use gmsh to create tet mesh
#     cmd = 'gmsh -3 -o '+outtet+' '+geofile
#     run_cmd(cmd,'gmsh failed, is it in your path?') 
    
#     # cleanup
#     cmd ='rm '+geofile
#     run_cmd(cmd,'rm temp geofile failed?') 
#     cmd='rm '+surftemp_stl
#     run_cmd(cmd,'rm temp stl surface failed?') 

#     # return tetmesh filename
#     return outtet

# calculate the eigenmodes
def eigenmodes(sdir, sid, surf, outev, outdir, num_modes):
    fem = Solver(surf)
    ev = fem(k=num_modes)
    
    evals = ev['Eigenvalues']
    emodes = ev['Eigenvectors']
    
    # remove the medial wall
    medial_wall = find_medial_wall(sdir, sid)
    emodes[medial_wall] = np.nan
    
    outev = os.path.join(outdir, outev)
    
    np.savetxt(outev, emodes)

# remove the medial wall
def find_medial_wall(sdir, sid):
    label = os.path.join(sdir, sid, 'label', 'lh.aparc.a2009s.annot')
    labels = read_label(label)

    indices = np.where(labels==1644825) # fs medial wall labels
    
    return indices

if __name__=="__main__":
    # Command Line options and error checking done here
    options = options_parse()

    my_print(options.label)
    my_print(options.surf)

    # if options.asegid is not None:
    #     surf = get_aseg_surf(options.sdir,options.sid,options.asegid,options.outsurf)
    #     outsurf = surf
    # elif options.label is not None:
    #     surf = get_label_surf(options.sdir,options.sid,options.label,options.surf,options.source,options.outsurf)
    #     outsurf = surf
    # elif options.aparcid is not None:
    #     surf = get_aparc_surf(options.sdir,options.sid,options.surf,options.aparcid,options.outsurf)    
    #     outsurf = surf
    if options.surf is not None:
        surf = make_surf(options.sdir,options.sid,options.surf)
        outsurf = options.outsurf
    #my_print(surf)
    #sys.exit(0)
    if surf is None:
        my_print('ERROR: no surface was created/selected?')
        sys.exit(1)

    # if options.dotet:
    #     #convert to tetmesh
    #     get_tetmesh(surf,options.outtet,options.fixiter)
    #     #run shapedna tetra
    #     calc_eigenmodes(options.outtet,options.outev,options)
    
    eigenmodes(options.sdir, options.sid, surf,options.outev,options.outdir, options.num)

    # always exit with 0 exit code
    sys.exit(0)
    
