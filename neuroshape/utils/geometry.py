from nipype.interfaces.freesurfer import MRIMarchingCubes
from neuroshape.utils.checks import is_string_like
#import gmsh
from lapy import TriaMesh
import warnings
from collections import OrderedDict

#gmsh.initialize()

import numpy as np
"""
Read and write geometry into different formats

    - Runs mri_mc from Freesurfer to create 2d surface
    - Projects 2d surface to gmsh and creates a tetrahedral mesh for LaPy
    - Writes out geometry in tetrahedral format or in Freesurfer binary
    - Writes out label files

Code was taken from nibabel.freesurfer package (https://github.com/nipy/nibabel/blob/master/nibabel/freesurfer/io.py).
"""

def mri_mc(in_file, label_value):
    """
    Runs class nipype.interfaces.freesurfer.MRIMarchingCubes()

    Parameters
    ----------
    in_file : str
        filename of label volume
    label_value : int
        label or mask value to compute marching cubes algorithm on
    out_file : str
        filename of output

    Returns
    -------
    str
        Name of the output file
        
    """
    
    mc = MRIMarchingCubes()
    mc.inputs.in_file = in_file
    mc.inputs.label_value = label_value
    mc.run()
    
    return mc.output_spec.out_file

def combine_geometry(in_file, label_values):
    geom = np.array()
    for label in label_values:
        if not type(label) == int:
            return ValueError('incorrect format for label inputs')
        
        out_file = mri_mc(in_file, label)
        geom = np.append(read_geometry(out_file))
    
    tria = TriaMesh(geom)
    
    return tria

# def write_geometry(model, outfile):
#     if not is_string_like(outfile):
#         return ValueError('incorrect format for filename output')
#     if not outfile[:-4] == ".msh":
#         outfile += ".msh"
    
#     gmsh.write(outfile)
#     gmsh.finalize()
    
#     return print(f'mesh file saved to {outfile}')
        
def convert_geometry(in_file, label_value, out_file=None):
    """
    Converts label or mask file to triangular mesh of class lapy.TriaMesh()

    Parameters
    ----------
    in_file : str
        filename of label volume
    label_value : int
        label or mask value to compute marching cubes algorithm on
    out_file : str, optional
        filename of output

    Returns
    -------
    tria : lapy.TriaMesh class

    Raises
    ------
    ValueError : inputs are incorrectly formatted

    """
    if not is_string_like(in_file) or not is_string_like(out_file):
        return ValueError(f'expected str and str, got {type(in_file)}, {type(out_file)}')
    if not type(label_value) == int:
        return ValueError(f'label must be an integer, got {type(label_value)}')
    
    if out_file is not None:
        mri_mc(in_file, label_value, out_file)
    else:
        out_file = mri_mc(in_file, label_value)
        
    coords, faces = read_geometry(out_file)
    
    tria = TriaMesh(coords, faces)
    
    return tria
    
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
    
def read_geometry(filepath, read_metadata=False, read_stamp=False):
    """Read a triangular format Freesurfer surface mesh.
    Parameters
    ----------
    filepath : str
        Path to surface file.
    read_metadata : bool, optional
        If True, read and return metadata as key-value pairs.
        Valid keys:
        * 'head' : array of int
        * 'valid' : str
        * 'filename' : str
        * 'volume' : array of int, shape (3,)
        * 'voxelsize' : array of float, shape (3,)
        * 'xras' : array of float, shape (3,)
        * 'yras' : array of float, shape (3,)
        * 'zras' : array of float, shape (3,)
        * 'cras' : array of float, shape (3,)
    read_stamp : bool, optional
        Return the comment from the file
    Returns
    -------
    coords : numpy array
        nvtx x 3 array of vertex (x, y, z) coordinates.
    faces : numpy array
        nfaces x 3 array of defining mesh triangles.
    volume_info : OrderedDict
        Returned only if `read_metadata` is True.  Key-value pairs found in the
        geometry file.
    create_stamp : str
        Returned only if `read_stamp` is True.  The comment added by the
        program that saved the file.
    """
    volume_info = OrderedDict()

    TRIANGLE_MAGIC = 16777214

    with open(filepath, "rb") as fobj:
        magic = _fread3(fobj)

        if magic == TRIANGLE_MAGIC:  # Triangle file
            create_stamp = fobj.readline().rstrip(b"\n").decode("utf-8")
            test_dev = fobj.peek(1)[:1]
            if test_dev == b"\n":
                fobj.readline()
            vnum = np.fromfile(fobj, ">i4", 1)[0]
            fnum = np.fromfile(fobj, ">i4", 1)[0]
            coords = np.fromfile(fobj, ">f4", vnum * 3).reshape(vnum, 3)
            faces = np.fromfile(fobj, ">i4", fnum * 3).reshape(fnum, 3)

            if read_metadata:
                volume_info = _read_volume_info(fobj)
        else:
            raise ValueError(
                "File does not appear to be a Freesurfer surface (triangle file)"
            )

    coords = coords.astype(float)  # XXX: due to mayavi bug on mac 32bits

    ret = (coords, faces)
    if read_metadata:
        if len(volume_info) == 0:
            warnings.warn("No volume information contained in the file")
        ret += (volume_info,)
    if read_stamp:
        ret += (create_stamp,)

    return ret

def write_label(filepath, vertices, values=None):
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