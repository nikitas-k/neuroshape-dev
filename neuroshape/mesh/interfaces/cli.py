"""
Interfaces with command-line interface, terminal, path, stdout, print messages
"""

import optparse
import sys
import os
import tempfile
import warnings
import errno
import subprocess
import uuid
import glob
import platform

warnings.filterwarnings('ignore', '.*negative int.*')
supported_os = ['Linux', 'Windows', 'Darwin']

def m_print(message):
    """
    print message, then flush stdout
    """
    print(message)
    sys.stdout.flush()
    
def split_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
    
def split_filename(obj):
    """
    Returns filename split into path, basename, and extension.

    Parameters
    ----------
    obj : str
        Filename

    Returns
    -------
    tuple of 3 str (path, basename, ext)

    """
    folder, basename = os.path.split(obj)
    basename, ext = os.path.splitext(basename)
    return folder, basename, ext

def _check_os():
    os_name = platform.system()
    if os_name not in supported_os:
        raise Exception(f'You are using an unsupported operating system: {os_name}')

def freesurfer_subjects_dir():
    """
    Returns the freesurfer environment variable $SUBJECTS_DIR, if it exists.

    Returns
    -------
    subjects_dir : str
        Path to Freesurfer subjects directory

    """
    subjects_dir = os.getenv('SUBJECTS_DIR')
    
    if subjects_dir is not None:
        return subjects_dir
    else:
        raise ValueError('Freesurfer subjects directory is not set!')
    

def matlab_path():
    """
    Returns the matlab path if it exists, otherwise tries to source it.

    Parameters
    ----------
    None.

    Returns
    -------
    str
        Path to matlab binary/executable if one can be found.

    """
    
    os_name = _check_os()
    if os_name in ['Linux', 'Darwin']:
        for dir in os.environ['PATH'].split(os.pathsep):
            if os.path.exists(os.path.join(dir, 'matlab')):
                return dir
            elif os_name == 'Darwin':
                if glob.glob('/Applications/MATLAB*') is not None:
                    folder = glob.glob('/Applications/MATLAB*')
                    if os.path.exists(os.path.join(folder, 'bin', 'matlab')):
                        return os.path.join(folder, 'bin', 'matlab')
                    else:
                        raise Warning('Check matlab installation')
                else:
                    raise Warning('matlab installation not in standard directory, please source')
        return None
    else: #windows
        for dir in os.environ['PATH'].split(os.pathsep):
            if os.path.exists(os.path.join(dir, 'matlab.exe')):
                return dir
        if glob.glob(r'C:\\Program Files\MATLAB*') is not None:
            folder = glob.glob(r'C:\\Program Files\MATLAB*')
            if os.path.exists(os.path.join(folder, 'bin', 'matlab.exe')):
                return os.path.join(folder, 'bin', 'matlab.exe')
            else:
                raise Warning('Check matlab installation')
        else:
            raise Warning('matlab installation not in standard directory, please source')

def check_write_access(folder):
    try:
        testfile = tempfile.TemporaryFile(dir = folder)
        testfile.close()
        return True
    except:
        return False
    



