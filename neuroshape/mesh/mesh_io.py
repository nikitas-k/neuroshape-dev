"""
meshio - high level read/write functions for different formats

High level mesh IO operations. Computes surface (and volume) models 
of different meshes including FreeSurfer subcortical and cortical structures
and saves in different formats.

@author - Nikitas C. Koussis
"""

import warnings
import os
import tempfile

import numpy as np

from .iosupport import ( 
    _select_reader, _uncompress, _select_writer, 
    is_string_like,
    )

warnings.filterwarnings('ignore', '.*negative int.*')
os.environ['OMP_NUM_THREADS'] = '1'

supported_types = ['ply', 'vtp', 'vtk', 'asc', 'fs', 'gii']
supported_formats = ['binary', 'ascii']

def read_surface(obj, itype=None):
    """Read surface data.

    See `itype` for supported file types.

    Parameters
    ----------
    obj : str
        Input filename.
    itype : {'ply', 'vtp', 'vtk', 'fs', 'asc', 'gii'}, optional
        Input file type. If None, it is deduced from `obj`. Files compressed
        with gzip (.gz) are also supported. Default is None.

    Returns
    -------
    output : tuple of np.ndarrays of shape (num_verts, 3), (num_faces, 3)

    Notes
    -----
    Function can read FreeSurfer geometry data in binary ('fs') and ascii
    ('asc') format. Gifti surfaces can also be loaded if nibabel is installed.

    See Also
    --------
    :func:`write_surface`

    """
    if is_string_like(obj) is True:
        if itype is None:
            itype = obj.split('.')[-1]
    
        if itype == 'gz':
            extension = obj.split('.')[-2]
            tmp = tempfile.NamedTemporaryFile(suffix='.' + extension, delete=False)
            tmp_name = tmp.name
            # Close and reopen because windows throws permission errors when both
            # reading and writing.
            tmp.close()
            _uncompress(obj, tmp_name)
            result = read_surface(tmp_name, extension)
            os.unlink(tmp_name)
            return result
    
        reader = _select_reader(itype)
        if itype == 'asc':
            return reader(obj, is_ascii=True)
        elif itype in ['pial', 'midthickness', 'inflated', 'white']:
            return reader(obj, is_ascii=False)
        else:
            return reader(obj)
    
    else:
        return ValueError("Input object must be a string")

def write_surface(obj, filename, oformat=None, otype=None):
    """Write surface data.

    See `otype` for supported file types.

    Parameters
    ----------
    obj : shape class
        Input pointset and data
    filename : str
        Output filename.
    oformat : {'ascii', 'binary'}, optional
        File format. Defaults to writer's default format.
        Only used when writer accepts format. Default is None.
    otype : {'ply', 'vtp', 'vtk', 'fs', 'asc', 'gii'}, optional
        File type. If None, type is deduced from `opth`. Default is None.

    Notes
    -----
    Function can save data in FreeSurfer binary ('fs') and ascii ('asc')
    format. Gifti surfaces can also be saved if nibabel is installed.

    See Also
    --------
    :func:`read_surface`

    """
    if otype is None:
        otype = filename.split('.')[-1]

    writer = _select_writer(otype)

    if otype not in ['vtp', 'tri', 'gii', 'obj']:
        if oformat == 'ascii' or otype == 'asc':
            writer(obj, is_ascii=True)
        else:
            writer(obj, filename)
    else:
        writer(obj, filename)


def convert_surface(obj, opth, itype=None, otype=None, oformat=None):
    """Convert between file types.

    Parameters
    ----------
    obj : str
        Input filename.
    opth : str
        Output filename.
    itype : str, optional
        Input file type. If None, type is deduced from input filename's
        extension. Default is None.
    otype : str, optional
        Output file type. If None, type is deduced from output filename's
        extension. Default is None.
    oformat : {'ascii', 'binary'}
        Output file format. Defaults to writer's default format.
        Only used when writer accepts format. Default is None.

    """
    reader = read_surface(obj, itype=itype, return_data=False, update=False)
    write_surface(reader, opth, oformat=oformat, otype=otype)
    

def read_data(obj, itype=None, iformat=None):
    """
    Read input map, must have same number of points as surface has vertices

    Parameters
    ----------
    obj : str
        Input filename
    itype : str, optional
        Input file type. If None, type is deduced from input filename's
        extension. Default is None.
    iformat : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    np.ndarray of shape=(num_verts,)

    """
    if is_string_like(obj) is True:
        if itype is None:
            itype = obj.split('.')[-1]
    
        if itype == 'gz':
            extension = obj.split('.')[-2]
            tmp = tempfile.NamedTemporaryFile(suffix='.' + extension, delete=False)
            tmp_name = tmp.name
            # Close and reopen because windows throws permission errors when both
            # reading and writing.
            tmp.close()
            _uncompress(obj, tmp_name)
            result = read_surface(tmp_name, extension)
            os.unlink(tmp_name)
            return result
        
        if itype == 'txt':
            return np.loadtxt(obj)
        
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
        
    if isinstance(obj, np.ndarray):
        if obj.ndim > 2:
            raise ValueError('Cannot import data array with more than two dimensions')
        
    return data

def write_data(obj, filename, otype=None, oformat=None):
    """
    Write surface map, output format is guessed from extension unless otype
    is specified.

    Parameters
    ----------
    obj : Shape.data attribute of np.ndarray or np.ndarray
        Data to write out.
    filename : str
        Output filename.
    oformat : {'ascii', 'binary'}, optional
        File format. Defaults to writer's default format.
        Only used when writer accepts format. Default is None.
    otype : {'ply', 'vtp', 'vtk', 'fs', 'asc', 'gii'}, optional
        File type. If None, type is deduced from `opth`. Default is None.
        
    """
    if otype is None:
        otype = obj.split('.')[-1]
        
    if otype in ['vtp', 'tri', 'gii', 'obj', 'asc']:
        writer = _select_writer(otype)
        if otype not in ['vtp', 'tri', 'gii', 'obj']:
            if oformat == 'ascii' or otype == 'asc':
                writer(obj, is_ascii=True)
            else:
                writer(obj, filename)
        else:
            writer(obj, filename)
    
    else:
        try:
            np.savetxt(filename, obj)

        except Exception as e:
            raise f'Error: {e}'