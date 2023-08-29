"""
IO support functions
"""

import tempfile
import gzip
import os
import warnings
import re
from collections import OrderedDict
import numpy as np
from ants import image_read, registration, apply_transforms
from neuromaps.datasets.atlases import fetch_mni152
from . import mesh_io
from .interfaces.cli import split_filename

# from .geometry import (
#     make_surface, make_aseg_surf, make_label_surf, find_medial_wall,
#     )

from vtk import (
    vtkPLYReader, vtkPLYWriter, vtkXMLPolyDataReader, vtkXMLPolyDataWriter,
    vtkPolyDataReader, vtkPolyDataWriter,
    )


import nibabel as nib
INTENT_VERTS = nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']
INTENT_FACES = nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']
INTENT_VERTDATA = nib.nifti1.intent_codes['NIFTI_INTENT_ESTIMATE']
has_nibabel = True

    
TRIANGLE_MAGIC = 16777214
QUAD_MAGIC = 16777215
NEW_QUAD_MAGIC = 16777213

_ANNOT_DT = ">i4"
"""Data type for Freesurfer `.annot` files.

Used by :func:`read_annot` and :func:`write_annot`.  All data (apart from
strings) in an `.annot` file is stored as big-endian int32.
"""

mni152_2mm = np.asarray(
                 [[  2.,    0.,   0.,    -90,],
                  [ -0.,    2.,   0.,   -126,],
                  [ -0.,    0.,   2.,    -72,],
                  [  0.,    0.,   0.,      1.]])

mni152_1mm = np.asarray(
                [[  -1.,    0.,    0.,   90.],
                 [   0.,    1.,    0., -126.],
                 [   0.,    0.,    1.,  -72.],
                 [   0.,    0.,    0.,    1.]])

def _uncompress(obj, opth, block_size=65536):
    """Uncompresses files. Currently only supports gzip.

    Parameters
    ----------
    obj : str
        Input filename.
    opth : str
        Output filename.
    block_size : int, optional
        Size of blocks of the input that are read at a time, by default 65536

    """
    if obj.split('.')[-1] == 'gz':
        with gzip.open(obj, 'rb') as i_file, open(opth, 'wb') as o_file:
            while True:
                block = i_file.read(block_size)
                if not block:
                    break
                else:
                    o_file.write(block)
    else:
        ValueError('Unknown file format')
        
def _select_reader(itype):
    if itype == 'ply':
        reader = vtkPLYReader()
    elif itype == 'vtp':
        reader = vtkXMLPolyDataReader()
    elif itype == 'vtk':
        reader = vtkPolyDataReader()
    elif itype in ['asc', 'pial', 'midthickness', 'inflated', 'white']:
        reader = read_fs
        if itype == 'asc':
            reader.SetFileTypeToASCII()
    elif itype == 'gii':
        reader = read_gifti
    # elif itype == 'nii':
    #     reader = read_nii()
    else:
        raise TypeError('Unknown input type \'{0}\'.'.format(itype))
    return reader

def _select_writer(otype):
    if otype == 'ply':
        writer = vtkPLYWriter()
    # elif otype == 'obj':
    #     writer = vtkOBJWriter()
    elif otype == 'vtp':
        writer = vtkXMLPolyDataWriter()
    elif otype == 'vtk':
        writer = vtkPolyDataWriter()
    elif otype in ['asc', 'pial', 'midthickness', 'inflated', 'white']:
        writer = write_fs
    elif otype == 'gii':
        writer = write_gifti
    # elif otype == 'nii':
    #     writer = write_nii()
    else:
        raise TypeError('Unknown output type \'{0}\'.'.format(otype))
    return writer

def get_ext(obj):
    """
    Get extension from filename in `obj`

    Parameters
    ----------
    obj : str
        Name of file to get extension.

    Returns
    -------
    str
        Extension of file.

    """
    if is_string_like(obj) is True:
        *_, ext = split_filename(obj)
        return ext
        
    else:
        raise ValueError('Input must be string')

def get_tkrvox2ras(voldim, voxres):
    """Generate transformation matrix to switch between tetrahedral and volume space.

    Parameters
    ----------
    voldim : array (1x3)
        Dimension of the volume (number of voxels in each of the 3 dimensions)
    voxres : array (!x3)
        Voxel resolution (resolution in each of the 3 dimensions)

    Returns
    ------
    T : array (4x4)
        Transformation matrix
    """

    T = np.zeros([4,4]);
    T[3,3] = 1;

    T[0,0] = -voxres[0];
    T[0,3] = voxres[0]*voldim[0]/2;

    T[1,2] = voxres[2];
    T[1,3] = -voxres[2]*voldim[2]/2;


    T[2,1] = -voxres[1];
    T[2,3] = voxres[1]*voldim[1]/2;

    return T
    
def is_string_like(obj):
    """ Check whether `obj` behaves like a string. """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True
    
def read_gifti(obj):
    data = None
    g = nib.load(obj)
    
    verts = g.get_arrays_from_intent(INTENT_VERTS)[0].data
    faces = g.get_arrays_from_intent(INTENT_FACES)[0].data
    try:
        data = g.get_arrays_from_intent(INTENT_VERTDATA)[0].data
    except:
        pass
    
    return verts, faces, data

def write_gifti(obj, filename, write_data=False):
    try:
        obj.v = obj.v
        obj.f = obj.f
    except:
        raise AttributeError('Object must be Shape class with `v` and `f` attributes')
        
    from nibabel.gifti.gifti import GiftiDataArray
    
    if not obj.polyshape == 3:
        raise ValueError('GIFTI writer only accepts triangular mesh')
    
    verts = GiftiDataArray(data=obj.v, intent=INTENT_VERTS)
    faces = GiftiDataArray(data=obj.f, intent=INTENT_FACES)
    if write_data is True:
        vertdata = GiftiDataArray(data=obj.data, intent=INTENT_VERTDATA)
        g = nib.GiftiImage(darrays=[verts, faces, vertdata])
    
    else:
        g = nib.GiftiImage(darrays=[verts, faces])
    
    nib.save(g, filename)
    
def read_fs(obj, is_ascii=False):
    surf = _read_geometry(obj, is_ascii)
    return surf

def write_fs(obj, filename):
    nib.save(obj, filename)
    
def read(obj):
    if is_string_like(obj) is True:
        itype = obj.split('.')[-1]
    
        if itype == 'gz':
            extension = obj.split('.')[-2]
            tmp = tempfile.NamedTemporaryFile(suffix='.' + extension, delete=False)
            tmp_name = tmp.name
            # Close and reopen because windows throws permission errors when both
            # reading and writing.
            tmp.close()
            _uncompress(obj, tmp_name)
            result = mesh_io.read_surface(tmp_name, extension)
            os.unlink(tmp_name)
            return result
        
        # if itype == 'nii':
        #     return read_nii(obj)
        
        reader = _select_reader(itype)
        if itype == 'asc':
            return reader(obj, is_ascii=True)
        else:
            return reader(obj)
        
    else: #just try to load it
        try:
            data = np.loadtxt(obj, dtype=float)
        
        except Exception as e:
            raise f'Error: {e}'
            
    return data
    
def _read_annot(filepath, orig_ids=False):
    with open(filepath, "rb") as fobj:
        dt = _ANNOT_DT

        # number of vertices
        vnum = np.fromfile(fobj, dt, 1)[0]

        # vertex ids + annotation values
        data = np.fromfile(fobj, dt, vnum * 2).reshape(vnum, 2)
        labels = data[:, 1]

        # is there a color table?
        ctab_exists = np.fromfile(fobj, dt, 1)[0]
        if not ctab_exists:
            raise Exception('Color table not found in annotation file')

        # in old-format files, the next field will contain the number of
        # entries in the color table. In new-format files, this must be
        # equal to -2
        n_entries = np.fromfile(fobj, dt, 1)[0]

        # We've got an old-format .annot file.
        if n_entries > 0:
            ctab, names = _read_annot_ctab_old_format(fobj, n_entries)
        # We've got a new-format .annot file
        else:
            ctab, names = _read_annot_ctab_new_format(fobj, -n_entries)

    # generate annotation values for each LUT entry
    ctab[:, [4]] = _pack_rgb(ctab[:, :3])

    if not orig_ids:
        ord = np.argsort(ctab[:, -1])
        mask = labels != 0
        labels[~mask] = -1
        labels[mask] = ord[np.searchsorted(ctab[ord, -1], labels[mask])]
    return labels, ctab, names

def _pack_rgb(rgb):
    """Pack an RGB sequence into a single integer.

    Used by :func:`read_annot` and :func:`write_annot` to generate
    "annotation values" for a Freesurfer ``.annot`` file.

    Parameters
    ----------
    rgb : ndarray, shape (n, 3)
        RGB colors

    Returns
    -------
    out : ndarray, shape (n, 1)
        Annotation values for each color.
    """
    bitshifts = 2 ** np.array([[0], [8], [16]], dtype=rgb.dtype)
    return rgb.dot(bitshifts)

def _read_annot_ctab_old_format(fobj, n_entries):
    """Read in an old-style Freesurfer color table from `fobj`.

    Note that the output color table ``ctab`` is in RGBT form, where T
    (transparency) is 255 - alpha.

    This function is used by :func:`read_annot`.

    Parameters
    ----------

    fobj : file-like
        Open file handle to a Freesurfer `.annot` file, with seek point
        at the beginning of the color table data.
    n_entries : int
        Number of entries in the color table.

    Returns
    -------

    ctab : ndarray, shape (n_entries, 5)
        RGBT colortable array - the last column contains all zeros.
    names : list of str
        The names of the labels. The length of the list is n_entries.
    """
    assert hasattr(fobj, 'read')

    dt = _ANNOT_DT
    # orig_tab string length + string
    length = np.fromfile(fobj, dt, 1)[0]
    orig_tab = np.fromfile(fobj, '>c', length)
    orig_tab = orig_tab[:-1]
    names = list()
    ctab = np.zeros((n_entries, 5), dt)
    for i in range(n_entries):
        # structure name length + string
        name_length = np.fromfile(fobj, dt, 1)[0]
        name = np.fromfile(fobj, "|S%d" % name_length, 1)[0]
        names.append(name)
        # read RGBT for this entry
        ctab[i, :4] = np.fromfile(fobj, dt, 4)

    return ctab, names

def _read_annot_ctab_new_format(fobj, ctab_version):
    """Read in a new-style Freesurfer color table from `fobj`.

    Note that the output color table ``ctab`` is in RGBT form, where T
    (transparency) is 255 - alpha.

    This function is used by :func:`read_annot`.

    Parameters
    ----------

    fobj : file-like
        Open file handle to a Freesurfer `.annot` file, with seek point
        at the beginning of the color table data.
    ctab_version : int
        Color table format version - must be equal to 2

    Returns
    -------

    ctab : ndarray, shape (n_labels, 5)
        RGBT colortable array - the last column contains all zeros.
    names : list of str
        The names of the labels. The length of the list is n_labels.
    """
    assert hasattr(fobj, 'read')

    dt = _ANNOT_DT
    # This code works with a file version == 2, nothing else
    if ctab_version != 2:
        raise Exception('Unrecognised .annot file version (%i)', ctab_version)
    # maximum LUT index present in the file
    max_index = np.fromfile(fobj, dt, 1)[0]
    ctab = np.zeros((max_index, 5), dt)
    # orig_tab string length + string
    length = np.fromfile(fobj, dt, 1)[0]
    np.fromfile(fobj, "|S%d" % length, 1)[0]  # Orig table path
    # number of LUT entries present in the file
    entries_to_read = np.fromfile(fobj, dt, 1)[0]
    names = list()
    for _ in range(entries_to_read):
        # index of this entry
        idx = np.fromfile(fobj, dt, 1)[0]
        # structure name length + string
        name_length = np.fromfile(fobj, dt, 1)[0]
        name = np.fromfile(fobj, "|S%d" % name_length, 1)[0]
        names.append(name)
        # RGBT
        ctab[idx, :4] = np.fromfile(fobj, dt, 4)

    return ctab, names

def _fread3(fobj):
    """Read a 3-byte int from an open binary file object
    Parameters
    ----------
    fobj : file
        File descriptor
    Returns
    -------
    n : int
        A 3 byte int
    """
    b1, b2, b3 = np.fromfile(fobj, ">u1", 3)
    return (b1 << 16) + (b2 << 8) + b3

def _fread3_many(fobj, n):
    """Read 3-byte ints from an open binary file object.
    Parameters
    ----------
    fobj : file
        File descriptor
    Returns
    -------
    out : 1D array
        An array of 3 byte int
    """
    b1, b2, b3 = np.fromfile(fobj, ">u1", 3 * n).reshape(-1, 3).astype(np.int64).T
    return (b1 << 16) + (b2 << 8) + b3

def _read_volume_info(fobj):
    """Helper for reading the footer from a surface file."""
    volume_info = OrderedDict()
    head = np.fromfile(fobj, ">i4", 1)
    if not np.array_equal(head, [20]):  # Read two bytes more
        head = np.concatenate([head, np.fromfile(fobj, ">i4", 2)])
        if not np.array_equal(head, [2, 0, 20]) and not np.array_equal(
            head, [2, 1, 20]
        ):
            warnings.warn("Unknown extension code.")
            return volume_info
        head = [2, 0, 20]

    volume_info["head"] = head
    for key in [
        "valid",
        "filename",
        "volume",
        "voxelsize",
        "xras",
        "yras",
        "zras",
        "cras",
    ]:
        pair = fobj.readline().decode("utf-8").split("=")
        if pair[0].strip() != key or len(pair) != 2:
            raise IOError("Error parsing volume info.")
        if key in ("valid", "filename"):
            volume_info[key] = pair[1].strip()
        elif key == "volume":
            volume_info[key] = np.array(pair[1].split()).astype(int)
        else:
            volume_info[key] = np.array(pair[1].split()).astype(float)
    # Ignore the rest
    return volume_info

def _read_geometry(obj, is_ascii=False):
    """Read a triangular format Freesurfer surface mesh.
    """
    if is_ascii:
        with open(obj) as fh:
            re_header = re.compile('^#!ascii version (.*)$')
            fname_header = re_header.match(fh.readline()).group(1)

            re_nverts_faces = re.compile('[\s]*(\d+)[\s]*(\d+)[\s]*$')
            re_n = re_nverts_faces.match(fh.readline())
            n_verts, n_faces = int(re_n.group(1)), int(re_n.group(2))

            x_verts = np.zeros((n_verts, 3))
            for i in range(n_verts):
                x_verts[i, :] = [float(v) for v in fh.readline().split()[:3]]

            x_faces = np.zeros((n_faces, 3), dtype=np.uintp)
            for i in range(n_faces):
                x_faces[i] = [np.uintp(v) for v in fh.readline().split()[:3]]

    else:
        with open(obj, 'rb') as fh:
            magic = _fread3(fh)
            if magic not in [TRIANGLE_MAGIC, QUAD_MAGIC, NEW_QUAD_MAGIC]:
                raise IOError('File does not appear to be a '
                              'FreeSurfer surface.')

            if magic in (QUAD_MAGIC, NEW_QUAD_MAGIC):  # Quad file
                n_verts, n_quad = _fread3(fh), _fread3(fh)

                (fmt, div) = ('>i2', 100) if magic == QUAD_MAGIC else ('>f4', 1)
                x_verts = np.fromfile(fh, fmt, n_verts * 3).astype(np.float64)
                x_verts /= div
                x_verts = x_verts.reshape(-1, 3)

                quads = _fread3_many(fh, n_quad * 4)
                quads = quads.reshape(n_quad, 4)
                n_faces = 2 * n_quad
                x_faces = np.zeros((n_faces, 3), dtype=np.uintp)

                # Face splitting follows (Remove loop in nib) -> Not tested!
                m0 = (quads[:, 0] % 2) == 0
                m0d = np.repeat(m0, 2)
                x_faces[m0d].flat[:] = quads[m0][:, [0, 1, 3, 2, 3, 1]]
                x_faces[~m0d].flat[:] = quads[~m0][:, [0, 1, 2, 0, 2, 3]]

            elif magic == TRIANGLE_MAGIC:  # Triangle file
                # create_stamp = fh.readline().rstrip(b'\n').decode('utf-8')
                fh.readline()
                fh.readline()

                n_verts, n_faces = np.fromfile(fh, '>i4', 2)
                x_verts = np.fromfile(fh, '>f4', n_verts * 3)
                x_verts = x_verts.reshape(n_verts, 3).astype(np.float64)

                x_faces = np.zeros((n_faces, 3), dtype=np.uintp)
                x_faces.flat[:] = np.fromfile(fh, '>i4', n_faces * 3)

    return (x_verts, x_faces)
    
def _read_label(filepath, read_scalars=False):
    """Load in a Freesurfer .label file.

    Parameters
    ----------
    filepath : str
        Path to label file.
    read_scalars : bool, optional
        If True, read and return scalars associated with each vertex.

    Returns
    -------
    label_array : numpy array
        Array with indices of vertices included in label.
    scalar_array : numpy array (floats)
        Only returned if `read_scalars` is True.  Array of scalar data for each
        vertex.
    """
    label_array = np.loadtxt(filepath, dtype=int, skiprows=2, usecols=[0])
    if read_scalars:
        scalar_array = np.loadtxt(filepath, skiprows=2, usecols=[-1])
        return label_array, scalar_array
    return label_array    

def _write_label(filepath, vertices, values=None):
    """
    Write Freesurfer label data `values` to filepath `filepath`

    Parameters
    ----------
    filepath : str
        String containing path to label file to be written
    vertices : ndarray, shape (n_vertices, 3)
        Coordinates of each vertex
    values : optional, shape (n_vertices,)
        Array of scalar data for each vertex. The default is None.
    """
    
    if values is not None:
        vector = np.asarray(values)
        vnum = np.prod(vector.shape)
        if vector.shape not in ((vnum,), (vnum, 1), (1, vnum), (vnum, 1, 1)):
            raise ValueError('Invalid shape: argument values must be a vector')
            
    else:
        vector = np.zeros(vnum, dtype=np.float32)
    
    start_line = '#!ascii label  , from subject  vox2ras=TkReg\n'
    magic_number = vnum + 6000
    
    with open(filepath, 'w') as fobj:
        fobj.write(start_line)
        fobj.write(f'{magic_number}\n')
        
        #now write vertex and label array
        label_array = np.vstack((np.array(range(vnum)), vertices.T, vector.T)).T.astype('>f4')
        np.savetxt(fobj, label_array, fmt=['%i','%f','%f','%f','%f'])
        
def _read_morph_data(filepath):
    """Read a Freesurfer morphometry data file.

    This function reads in what Freesurfer internally calls "curv" file types,
    (e.g. ?h. curv, ?h.thickness), but as that has the potential to cause
    confusion where "curv" also refers to the surface curvature values,
    we refer to these files as "morphometry" files with PySurfer.

    Parameters
    ----------
    filepath : str
        Path to morphometry file

    Returns
    -------
    curv : numpy array
        Vector representation of surface morpometry values
    """
    with open(filepath, "rb") as fobj:
        magic = _fread3(fobj)
        if magic == 16777215:
            vnum = np.fromfile(fobj, ">i4", 3)[0]
            curv = np.fromfile(fobj, ">f4", vnum)
        else:
            vnum = magic
            _fread3(fobj)
            curv = np.fromfile(fobj, ">i2", vnum) / 100
    return curv

# def read_nii(filepath, mask=None):
#     """read nifti and create surface using optional mask"""
    
#     return data
    
# def write_nii(filepath):
#     """write surface as nii"""
    
    

def _check_mni(in_file):
    """
    Checks if input image is in MNI152 space
    """
    
    img = nib.load(in_file)
    
    if img.affine != mni152_2mm:
        if img.affine != mni152_1mm:
            return False
        
    else:
        return True
    
def native_to_mni152(in_file, nonlinear=True):
    """
    Linear or nonlinear registration of native volumetric image to MNI152 space
    Uses ANTsPy
    """
    
    img = image_read(in_file)
    
    # get template image
    mni_file = fetch_mni152(density='1mm').get('2009cAsym_T1w')
    mni = image_read(mni_file)

    if nonlinear is True:
        transform_type='SyN'
        
    else:
        transform_type='Affine'
        
    # do transform
    fixed_image = mni
    moving_image = img
    
    mytx = registration(fixed=fixed_image, moving=moving_image, type_of_transform=transform_type)
    
    warped_moving_image = apply_transforms(fixed=fixed_image, moving=moving_image,
                                           transformlist=mytx['fwdtransforms'])
    
    # rebuild as nib.Nifti1Image
    transformed_image = warped_moving_image.to_nibabel()
    
    return transformed_image

def create_temp_surface(surface_input, surface_output_filename):
    """Write surface to a new vtk file.

    Parameters
    ----------
    surface_input : brainspace compatible object
        Loaded vtk object corresponding to a surface triangular mesh
    surface_output_filename : str
        Filename of surface to be saved
    """

    f = open(surface_output_filename, 'w')
    f.write('# vtk DataFile Version 2.0\n')
    f.write(surface_output_filename + '\n')
    f.write('ASCII\n')
    f.write('DATASET POLYDATA\n')
    f.write('POINTS ' + str(np.shape(surface_input.Points)[0]) + ' float\n')
    for i in range(np.shape(surface_input.Points)[0]):
        f.write(' '.join(map(str, np.array(surface_input.Points[i, :]))))
        f.write('\n')
    f.write('\n')
    f.write('POLYGONS ' + str(np.shape(surface_input.polys2D)[0]) + ' ' + str(4* np.shape(surface_input.polys2D)[0]) + '\n')
    for i in range(np.shape(surface_input.polys2D)[0]):
        f.write(' '.join(map(str, np.append(3, np.array(surface_input.polys2D[i, :])))))
        f.write('\n')
    f.close()
    
def get_indices(surface_original, surface_new):
    """Extract indices of vertices of the two surfaces that match.

    Parameters
    ----------
    surface_original : brainspace compatible object
        Loaded vtk object corresponding to a surface triangular mesh
    surface_new : brainspace compatible object
        Loaded vtk object corresponding to a surface triangular mesh

    Returns
    ------
    indices : array
        indices of vertices
    """

    indices = np.zeros([np.shape(surface_new.Points)[0],1])
    for i in range(np.shape(surface_new.Points)[0]):
        indices[i] = np.where(np.all(np.equal(surface_new.Points[i,:],surface_original.Points), axis=1))[0][0]
    indices = indices.astype(int)
    
    return indices