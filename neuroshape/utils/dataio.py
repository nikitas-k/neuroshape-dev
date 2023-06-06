from .checks import is_string_like
from pathlib import Path
import numpy as np
import nibabel as nib

import os
import os.path as op
import subprocess
import re

__all__ = ['dataio',
           'load',
           'save',
           'run']

def dataio(x):
    """
    Flexible data I/O for core classes. Adapted from brainsmash.utils.dataio
    by Nikitas C. Koussis 2023
    
    To facilitate user inputs, this function loads data from:
        - lists
        - neuroimaging files
        - txt files
        - npy files
        - array_like data

    Parameters
    ----------
    x : filename(s) or np.ndarray or np.memmap

    Returns
    -------
    np.ndarray or np.memmap
    
    Raises
    ------
    FileExistsError : file does not exist
    RuntimeError : file is empty
    ValueError : file type cannot be determined or is not implemented
    TypeError : input is not a filename or array_like object

    """
    if type(x) == list:
        return np.row_stack(
            (load(x_) for x_ in x
             )
            )
    elif is_string_like(x):
        if not Path(x).exists():
            raise FileExistsError("file does not exist: {}".format(x))
        if Path(x).stat().st_size == 0:
            raise RuntimeError("file is empty: {}".format(x))
        if Path(x).suffix == ".npy":  # memmap
            return np.load(x, mmap_mode='r')
        if Path(x).suffix == ".txt":  # text file
            return np.loadtxt(x).squeeze()
        try:
            return load(x)
        except TypeError:
            raise ValueError(
                "expected npy or txt or gii file, got {}".format(
                    Path(x).suffix))
    else:
        if not isinstance(x, np.ndarray):
            raise TypeError(
                "expected filename or array_like obj, got {}".format(type(x)))
        return x
        
    
def load(x):
    """
    Load data contained in a GIFTI-format neuroimaging file.

    Parameters
    ----------
    filename : filename
        Path to neuroimaging file

    Returns
    -------
    array : np.ndarray or tuple of np.ndarrays
        Brain map data stored in `x`. Must be a surface file.
        
    Raises
    ------
    ValueError : `x` is not a surface file or invalid input (e.g., two hemispheres)
    TypeError : `x` has unknown filetype

    """
    
    try:
        array = nib.load(x).agg_data()
    except AttributeError:
        raise TypeError("This file cannot be loaded: {}".format(x)) 
        
    return array

"""
Utility function
Taken from neuromaps/utils
"""

def run(cmd, env=None, return_proc=False, quiet=False, **kwargs):
    """
    Runs `cmd` via shell subprocess with provided environment `env`

    Parameters
    ----------
    cmd : str
        Command to be run as single string
    env : dict, optional
        If provided, dictionary of key-value pairs to be added to base
        environment when running `cmd`. Default: None
    return_proc : bool, optional
        Whether to return CompletedProcess object. Default: false
    quiet : bool, optional
        Whether to suppress stdout/stderr from subprocess. Default: False

    Returns
    -------
    proc : subprocess.CompletedProcess
        Process output

    Raises
    ------
    subprocess.CalledProcessError
        If subprocess does not exit cleanly

    Examples
    --------
    >>> from neuromaps import utils
    >>> p = utils.run('echo "hello world"', return_proc=True, quiet=True)
    >>> p.returncode
    0
    >>> p.stdout  # doctest: +SKIP
    'hello world\\n'
    """

    merged_env = os.environ.copy()
    if env is not None:
        if not isinstance(env, dict):
            raise TypeError('Provided `env` must be a dictionary, not {}'
                            .format(type(env)))
        merged_env.update(env)

    opts = dict(check=True, shell=True, universal_newlines=True)
    opts.update(**kwargs)
    if quiet:
        opts.update(dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE))

    try:
        proc = subprocess.run(cmd, env=merged_env, **opts)
    except subprocess.CalledProcessError as err:
        raise subprocess.SubprocessError(
            f'Command failed with non-zero exit status {err.returncode}. '
            f'Error traceback: "{err.stderr.strip()}"'
        )

    if return_proc:
        return proc
    

def save(x, filename, outdir=None, prefix="", suffix="", ext=None):
    """
    Write data using nib.save() method of `x` contained in a 
    nib.gifti.gifti.GiftiImage class.

    Parameters
    ----------
    x : nib.gifti.gifti.GiftiImage class or np.ndarray
        
    filename : str
        Path to input file
        
    suffix : str
        Suffix to add to the `basename`. (default is '')
        
    ext : str
        Extension to use for the new filename.

    Returns
    -------
    out : str
        Path to output file

    """
    
    out = fname_presuffix(filename, outdir, prefix, suffix, ext)
    
    if isinstance(x, np.ndarray):
        X = nib.GiftiImage()
        X.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(x.astype('float32')))
        nib.save(X, out)
    
    elif isinstance(x, nib.gifti.gifti.GiftiImage):
        nib.save(x, out)
        
    else:
        return ValueError('img input to save() must be nib.GiftiImage class or'
                          ' np.ndarray')
    
    return out
        
def _gen_filename(name, outdir=None, suffix="", ext=None):
    """Generate a filename based on the given parameters.
    The filename will take the form: <basename><suffix><ext>.
    Parameters
    ----------
    name : str
        Filename to base the new filename on.
    suffix : str
        Suffix to add to the `basename`.  (defaults is '' )
    ext : str
        Extension to use for the new filename.
    Returns
    -------
    fname : str
        New filename based on given parameters.
    """
    if not name:
        raise ValueError("Cannot generate filename - filename not set")

    fpath, fname, fext = split_filename(name)
    if ext is None:
        ext = fext
    if outdir is None:
        if fpath is None:
            outdir = "."
        else:
            outdir = fpath
    return op.join(outdir, fname + suffix + ext)

def split_filename(fname):
    """Split a filename into parts: path, base filename and extension.

    Parameters
    ----------
    fname : str
        file or path name

    Returns
    -------
    pth : str
        base path from fname
    fname : str
        filename from fname, without extension
    ext : str
        file extension from fname

    Examples
    --------
    >>> from nipype.utils.filemanip import split_filename
    >>> pth, fname, ext = split_filename('/home/data/subject.nii.gz')
    >>> pth
    '/home/data'

    >>> fname
    'subject'

    >>> ext
    '.nii.gz'

    """

    special_extensions = [".nii.gz", ".tar.gz", ".niml.dset"]
    gifti_extensions = [".gii.gz", ".gii"]

    pth = op.dirname(fname)
    fname = op.basename(fname)

    ext = None            
    for special_ext in special_extensions:
        ext_len = len(special_ext)
        if (len(fname) > ext_len) and (fname[-ext_len:].lower() == special_ext.lower()):
            ext = fname[-ext_len:]
            fname = fname[:-ext_len]
            break
    if not ext:
        fname, ext = op.splitext(fname)

    return pth, fname, ext

def fname_presuffix(fname, outdir=None, prefix="", suffix="", newpath=None, use_ext=True):
    """Manipulates path and name of input filename

    Parameters
    ----------
    fname : string
        A filename (may or may not include path)
    
    prefix : string
        Characters to prepend to the filename
    suffix : string
        Characters to append to the filename
    newpath : string
        Path to replace the path of the input fname
    use_ext : boolean
        If True (default), appends the extension of the original file
        to the output name.

    Returns
    -------
    Absolute path of the modified filename

    >>> from nipype.utils.filemanip import fname_presuffix
    >>> fname = 'foo.nii.gz'
    >>> fname_presuffix(fname,'pre','post','/tmp')
    '/tmp/prefoopost.nii.gz'

    >>> from nipype.interfaces.base import Undefined
    >>> fname_presuffix(fname, 'pre', 'post', Undefined) == \
            fname_presuffix(fname, 'pre', 'post')
    True

    """
    pth, fname, ext = split_filename(fname)
    if outdir:
        pth = outdir
    if not use_ext:
        ext = ""

    # No need for isdefined: bool(Undefined) evaluates to False
    if newpath:
        pth = op.abspath(newpath)
    return op.join(pth, prefix + fname + suffix + ext)


def _fnames_presuffix(fnames, prefix="", suffix="", newpath=None, use_ext=True):
    """Calls fname_presuffix for a list of files."""
    f2 = []
    for fname in fnames:
        f2.append(_fname_presuffix(fname, prefix, suffix, newpath, use_ext))
    return f2

