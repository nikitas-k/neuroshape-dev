import numpy as np
from joblib import Parallel, delayed
from brainsmash.utils.dataio import dataio
import optparse

from neuroshape.utils.checks import is_string_like
from neuroshape.utils.eigen import (
    compute_eigenmodes, 
    maximise_recon_metric,
    eigen_decomposition,
    )
from neuromaps.stats import compare_images
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm
import sys
import os
import tempfile
import errno

cmap = plt.get_cmap('viridis')

def m_print(message):
    """
    print message, then flush stdout
    """
    print(message)
    sys.stdout.flush()

HELPTEXT = """

SUMMARY

Attempts to reconstruct the given map `y` using LBO eigenmodes `emodes`.
If `emodes` do not exist, computes them.

Reconstructs the given map `y` by fitting a weighted sum of `emodes` in a
GLM-like fashion (i.e., `y` = sum( `beta` * `emodes` ) + error ), using either
LU decomposition to solve the normal equation, or if the solution cannot
be inverted then uses linear least-squares to calculate the coefficients
`beta`.

Uses the coefficients `beta` to reconstruct the original data `y` and provides
an output of reconstructions and reconstruction accuracies.

Input can be one of the following:
--metric  : a Connectome Workbench metric file, usually has extension '.func.gii'
--shape   : a Connectome Workbench shape file, usually has extension '.shape.gii'
--fssurf  : a FreeSurfer curvature, thickness, or local gyrification index file
--file    : a ascii file containing scalar values for each vertex

"""

TMPTXT="""
REQUIRED ARGUMENTS

--sid <name>       Subject ID

--surf <name>      A surface file, either in .vtk or .surf.gii format

One of the following:
    
--metric <name>    A Connectome Workbench metric file containing a scalar value
                   for every vertex in <surf>, with extension ".func.gii."

--shape <name>     A Connectome Workbench shape file containing a scalar value
                   for every vertex in <surf>, with extension ".shape.gii"

--file <name>      ASCII file with scalars on each line for every vertex in <surf>


OPTIONAL ARGUMENTS

--sdir <name>      Subjects directory (or set via environment $SUBJECTS_DIR)

--emodes <name>    ASCII file of precomputed eigenmodes (if None, compute eigenmodes on the surface in <surf>)

--outdir <name>    Output directory (default: <sdir>/<sid>/neuroshape/ )

--outevec <name>   Name for eigenmodes output if none is passed to emodes
                   (default: <surf>.<shapedna_parameters>.txt)

--outfile <name>   Name for file of reconstructed values 
                   (default: <surface_metric>.recon.<num>_modes.txt)
                   
--savegii <name>   Name for gifti metric output of reconstructed values
                   (default: <surface_metric>.recon.<num>_modes.shape.gii)
                   
--savenii <name>   Name for nifti metric output of reconstructed values
                   (default: <surface_metric>.recon.<num>_modes.nii)
                   Note: surface map is interpolated in MNI152 2mm space to
                   avoid upsampling.
                   
ShapeDNA parameters if --emodes is not passed (see shapeDNA [1] for details):
    
--num <int>        Number of eigenvalues/modes to compute (default: 50)
                   
--degree <int>     FEM degree (default 1)

--bcond <int>      Boundary condition (0=Dirichlet, 1=Neumann default)

--evec            Additionally compute eigenvectors

--ignorelq        Ignore low quality in input mesh
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
    h_outevec     = 'Full path for eigenvalue output (default: <outdir>/<(out)surf or outtet>.ev )'
    
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
    group.add_option('--outevec',   dest='outevec',    help=h_outevec)
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
        m_print('\nERROR: specify subjects directory via --sdir or $SUBJECTS_DIR\n')
        sys.exit(1)
        
    if options.sid is None:
        parser.print_help()
        m_print('\nERROR: Specify --sid\n')
        sys.exit(1)
        
    subjdir = os.path.join(options.sdir, options.sid)
    if not os.path.exists(subjdir):
        m_print('ERROR: cannot find sid in subjects directory\n')
        sys.exit(1)
        
    if options.label is not None and options.surf is None:
        parser.print_help()
        m_print('\nERROR: Specify --surf with --label\n')
        sys.exit(1)  
    if options.aparcid is not None and options.surf is None:
        parser.print_help()
        m_print('\nERROR: Specify --surf with --aparc\n')
        sys.exit(1)  
    # input needs to be either a surf or aseg label(s)
    if options.asegid is None and options.surf is None:
        parser.print_help()
        m_print('\nERROR: Specify either --asegid or --surf\n')
        sys.exit(1)
    # and it cannot be both
    if options.asegid is not None and options.surf is not None:
        parser.print_help()
        m_print('\nERROR: Specify either --asegid or --surf (not both)\n')
        sys.exit(1)  
    
    # set default output dir (maybe this should be ./ ??)
    if options.outdir is None:
        options.outdir = os.path.join(subjdir, 'eigenmodes')
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
        m_print('\nERROR: '+options.outdir+' not writeable (check access)!\n')
        sys.exit(1)
    
    # initialize outsurf
    if options.outsurf is None:
        # for aseg stuff, we need to create a surface (and we'll keep it around)
        if options.asegid is not None:
            astring  = '_'.join(options.asegid)
            #surfname = 'aseg.'+astring+'.surf'
            surfname = 'aseg.' + astring + '.vtk'
            options.outsurf = os.path.join(options.outdir,surfname)    
        elif options.label is not None:
            surfname = os.path.basename(options.surf) + '.' + os.path.basename(options.label) + '.vtk'
            options.outsurf = os.path.join(options.outdir, surfname)
        elif options.aparcid is not None:
            astring  = '_'.join(options.aparcid)
            surfname = os.path.basename(options.surf) + '.aparc.' + astring + '.vtk'
            options.outsurf = os.path.join(options.outdir, surfname)          
        else:
            # for surfaces, a refined/smoothed version could be written
            surfname = os.path.basename(options.surf) +' .vtk'
            options.outsurf = os.path.join(options.outdir, surfname)
    else:
        # make sure it is vtk ending
        if (os.path.splitext(options.outsurf)[1]).upper() != '.VTK':
            m_print('ERROR: outsurf needs vtk extension (VTK format)')
            sys.exit(1)
    
    # for 3d processing, initialize outtet:
    if options.dotet and options.outtet is None:
        surfname = os.path.basename(options.outsurf)
        options.outtet = os.path.join(options.outdir, surfname+ '.msh')    
    
    # set source to sid if empty
    if options.source is None:
        options.source = options.sid
        
    # if label does not exist, search in subject label dir
    if options.label is not None and not os.path.isfile(options.label):
        ltemp = os.path.join(options.sdir, options.source, 'label', options.label)
        if os.path.isfile(ltemp):
            options.label = ltemp
        else:
            parser.print_help()
            m_print('\nERROR: Specified --label not found\n')
            sys.exit(1)  
                
    # initialize outevec 
    if options.outevec is None:
        if options.dotet:
            options.outevec = options.outtet + '.txt'
        else:
            options.outevec = options.outsurf + '.txt'
        
    return options

# Gets global path to surface input (if it is a FS surf)
def get_path_surf(sdir, sid, surf):
    return os.path.join(sdir, sid, 'surf', surf)
   
if __name__== "__main__":
    # Command Line options and error checking done here
    options = options_parse()
    
    if options.surf is None:
        m_print('ERROR: no surface was created/selected?')
        sys.exit(1)

    m_print(options.label)
    m_print(options.surf)