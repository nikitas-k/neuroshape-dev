"""
Geometry functions
"""

import os
from .iosupport import _read_geometry

def normalize_surf():
    """Normalize tetrahedral surface.

    Parameters
    ----------
    tet : lapy compatible object
        Loaded vtk object corresponding to a surface tetrahedral mesh
    nifti_input_filename : str
        Filename of input volume where the relevant ROI have voxel values = 1
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
    tet_norm : lapy compatible object
        Loaded vtk object corresponding to the normalized surface tetrahedral mesh
    """

    nifti_input_file_head, nifti_input_file_tail = os.path.split(nifti_input_filename)
    nifti_input_file_main, nifti_input_file_ext = os.path.splitext(nifti_input_file_tail)

    ROI_number, ROI_volume = calc_volume(nifti_input_filename)

    # normalization process
    tet_norm = tet
    if normalization_type == 'number':
        tet_norm.v = tet.v/(ROI_number**(1/3))
    elif normalization_type == 'volume':
        tet_norm.v = tet.v/(ROI_volume**(1/3))
    elif normalization_type == 'constant':
        tet_norm.v = tet.v/(normalization_factor**(1/3))
    else:
        pass

    # writing normalized surface to a vtk file
    if normalization_type == 'number' or normalization_type == 'volume' or normalization_type == 'constant':
        surface_output_filename = nifti_input_filename + '_norm=' + normalization_type + '.tetra.vtk'

        f = open(surface_output_filename, 'w')
        f.write('# vtk DataFile Version 2.0\n')
        f.write(nifti_input_file_tail + '\n')
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')
        f.write('POINTS ' + str(np.shape(tet.v)[0]) + ' float\n')
        for i in range(np.shape(tet.v)[0]):
            f.write(' '.join(map(str, tet_norm.v[i, :])))
            f.write('\n')
        f.write('\n')
        f.write('POLYGONS ' + str(np.shape(tet.t)[0]) + ' ' + str(5 * np.shape(tet.t)[0]) + '\n')
        for i in range(np.shape(tet.t)[0]):
            f.write(' '.join(map(str, np.append(4, tet.t[i, :]))))
            f.write('\n')
        f.close()

    return tet_norm

    
def make_surface(obj, outsurf=None):
    """
    creates surface from freesurfer
    """    
    coords, faces = _read_geometry(get_path_surf(sdir, sid, surf))
    
    # get hemi from surf
    splitsurf = surf.split(".", 1)
    hemi = splitsurf[0]
    surf = splitsurf[1]
    
    if outsurf is None: # base case
        outsurf = os.path.join(outdir, hemi + '.' + surf + '.vtk')
    
    # save tria mesh to outsurf
    tria = TriaMesh(coords, faces)
    TriaIO.export_vtk(tria, outsurf)
    
    # return surf name
    return outsurf
    
def make_label_surf(sdir, sid, label, surf, source, outsurf):
    """
    creates tria surface from label (and specified surface)
    maps the label first if source is different from sid
    """
    subjdir  = os.path.join(sdir, sid)
    outdir   = os.path.dirname(outsurf)
    stlsurf  = os.path.join(outdir, 'label' + str(uuid.uuid4()) + '.stl')

    # get hemi from surf
    splitsurf = surf.split(".",1)
    hemi = splitsurf[0]
    surf = splitsurf[1]
    # map label if necessary
    mappedlabel = label
    if (source != sid):
        mappedlabel = os.path.join(outdir, os.path.basename(label) + '.' + str(uuid.uuid4()) + '.label')
        cmd = 'mri_label2label --sd ' + sdir + ' --srclabel ' + label + ' --srcsubject ' + source + ' --trgsubject ' + sid + ' --trglabel ' + mappedlabel + ' --hemi ' + hemi + ' --regmethod surface'
        subprocess.run(cmd)     
    # make surface path (make sure output is stl, this currently needs fsdev, will probably be in 6.0)
    cmd = 'label2patch -writesurf -sdir ' + sdir + ' -surf ' + surf + ' ' + sid + ' ' + hemi + ' ' + mappedlabel + ' ' + stlsurf
    subprocess.run(cmd)     
    cmd = 'mris_convert ' + stlsurf + ' ' + outsurf
    subprocess.run(cmd)
    
    # cleanup map label if necessary
    if (source != sid):
        cmd ='rm ' + mappedlabel
        subprocess.run(cmd)     
    cmd = 'rm ' + stlsurf
    os.system(cmd)
    
    # make and write surface
    outsurf = make_surf(sdir, sid, surf, outdir, outsurf)
    
    # return surf name
    return outsurf

def make_aseg_surf:
    """
    Creates a surface from the aseg and label info
    and writes it to the outdir
    """
    astring2 = ' '.join(asegid)
    subjdir  = os.path.join(sdir, sid)
    aseg     = os.path.join(subjdir, 'mri', 'aseg.mgz')
    norm     = os.path.join(subjdir, 'mri', 'norm.mgz')  
    outdir   = os.path.dirname(outsurf)    
    tmpname  = 'aseg.' + str(uuid.uuid4())
    segf     = os.path.join(outdir, tmpname + '.mgz')
    segsurf  = os.path.join(outdir, tmpname + '.vtk')
    # binarize on selected labels (creates temp segf)
    ptinput = aseg
    ptlabel = str(asegid[0])
    #if len(asegid) > 1:
    # always binarize first, otherwise pretess may scale aseg if labels are larger than 255 (e.g. aseg+aparc, bug in mri_pretess?)
    cmd ='mri_binarize --i ' + aseg + ' --match ' + astring2 + ' --o ' + segf
    subprocess.run(cmd) 
    ptinput = segf
    ptlabel = '1'
    # if norm exist, fix label (pretess)
    if os.path.isfile(norm):
        cmd ='mri_pretess ' + ptinput + ' ' + ptlabel + ' ' + norm + ' ' + segf
        subprocess.run(cmd) 
    else:
        if not os.path.isfile(segf):
            # cp segf if not exist yet
            # (it exists already if we combined labels above)
            cmd = 'cp ' + ptinput + ' ' + segf
            subprocess.run(cmd) 
    # runs marching cube to extract surface
    cmd ='mri_mc ' + segf + ' ' + ptlabel + ' ' + segsurf
    subprocess.run(cmd) 
    # convert to stl
    cmd ='mris_convert ' + segsurf + ' ' + outsurf
    subprocess.run(cmd)
    # cleanup temp files
    cmd ='rm ' + segf
    subprocess.run(cmd) 
    cmd ='rm ' + segsurf
    subprocess.run(cmd)
    
    
    # return surf name
    return outsurf

def make_aparc_surf(sdir, sid, surf, aparcid, outsurf):
    """
    Creates a surface from the aparc and label number
    and writes it to the outdir
    """
    astring2 = ' '.join(aparcid)
    subjdir  = os.path.join(sdir, sid)
    outdir   = os.path.dirname(outsurf)    
    rndname = str(uuid.uuid4()) 
    # get hemi from surf
    hemi = surf.split(".", 1)[0]
    # convert annotation id to label:
    alllabels = ''
    for aid in aparcid:
        # create label of this id
        outlabelpre = os.path.join(outdir, hemi + '.aparc.' + rndname)
        cmd = 'mri_annotation2label --sd ' + sdir + ' --subject ' + sid + ' --hemi ' + hemi + ' --label ' + str(aid) + ' --labelbase ' + outlabelpre 
        subprocess.run(cmd) 
        alllabels = alllabels + '-i ' + outlabelpre + "-%03d.label" % int(aid) + ' ' 
    # merge labels (if more than 1)
    mergedlabel = outlabelpre + "-%03d.label" % int(aid)
    if len(aparcid) > 1:
        mergedlabel = os.path.join(outdir, hemi + '.aparc.all.' + rndname + '.label')
        cmd = 'mri_mergelabels ' + alllabels + ' -o ' + mergedlabel
        subprocess.run(cmd) 
    # make to surface (call subfunction above)
    get_label_surf(sdir, sid, mergedlabel, surf, sid, outsurf)
    # cleanup
    if len(aparcid) > 1:
        cmd ='rm ' + mergedlabel
        subprocess.run(cmd)
    for aid in aparcid:
        outlabelpre = os.path.join(outdir, hemi + '.aparc.' + rndname + "-%03d.label" % int(aid))
        cmd ='rm ' + outlabelpre
        subprocess.run(cmd)
    # return surf name
    return outsurf
    
def find_medial_wall:
    label = os.path.join(sdir, sid, 'label', 'lh.aparc.a2009s.annot')
    labels, ctab, names = read_annot(label)

    indices = np.argwhere(labels==-1) # fs medial wall labels
    
    return indices
    
def nearest_neighbor(P, X, radius=None):
    """
    Find the one-nearest neighbors of vertices in points `P` on another 
    surface `X` using Delaunay triangulation and KDTree query.

    Parameters
    ----------
    P : np.ndarray of shape (N,3)
        Points to search for within the coordinate set of `X`. `P` can
        be a single point
    X : np.ndarray of shape (M,3)
        Vertices of the surface to search within
    radius : float
        Radius to search for nearest neighbors within

    Returns
    -------
    nearest_indexes : int
        Indexes of one-nearest neighbors of vertices in `P`. Note that
        if two vertices in `X` are the same distance away from a point in `P`,
        function returns only the first one.

    """
    
    # Create Delaunay triangulation for first surface
    tri = Delaunay(X)
    
    # Create tree of vertices to query on
    kdtree = KDTree(X)

    indices = np.empty(P.shape[0], dtype=int)
    for i, p in enumerate(P):
        simplex_index = tri.find_simplex(p)
        if simplex_index == -1 or (radius is not None and not _is_point_within_radius(p, X[tri.simplices[simplex_index]], radius)):
            _, nearest_neighbor_index = kdtree.query(p)
        else:
            simplex_vertices = X[tri.simplices[simplex_index]]
            dist = np.linalg.norm(simplex_vertices - p, axis=1)
            if radius is not None:
                valid_indices = np.where(dist <= radius)[0]
                if valid_indices.size == 0:
                    _, nearest_neighbor_index = kdtree.query(p)
                else:
                    nearest_neighbor_index = tri.simplices[simplex_index][valid_indices[np.argmin(dist[valid_indices])]]
            else:
                nearest_neighbor_index = tri.simplices[simplex_index][np.argmin(dist)]
        indices[i] = nearest_neighbor_index

    return indices


def _is_point_within_radius(p, vertices, radius):
    """
    Check if a point is within a given radius of any vertex in a set of vertices.
    """
    return np.any(np.linalg.norm(vertices - p, axis=1) <= radius)


def calc_volume(nifti_input_filename):
    """Calculate the physical volume of the ROI in the nifti file.

    Parameters
    ----------
    nifti_input_filename : str
        Filename of input volume where the relevant ROI have voxel values = 1

    Returns
    ------
    ROI_number : int
        Total number of non-zero voxels
    ROI_volume : float
        Total volume of non-zero voxels in physical dimensions   
    """

    # Load data
    ROI_data = nib.load(nifti_input_filename)
    roi_data = ROI_data.get_fdata()

    # Get voxel dimensions in mm
    voxel_dims = (ROI_data.header["pixdim"])[1:4]
    voxel_vol = np.prod(voxel_dims)

    # Compute volume
    ROI_number = np.count_nonzero(roi_data)
    ROI_volume = ROI_number * voxel_vol

    # print("Number of non-zero voxels = {}".format(ROI_number))
    # print("Volume of non-zero voxels = {} mm^3".format(ROI_volume))

    return ROI_number, ROI_volume