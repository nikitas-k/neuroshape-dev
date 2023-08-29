"""
MATLAB interfaces. All routines require matlab to be installed.
"""
import os
from .cli import matlab_path, get_ext, check_write_access
import tempfile
import uuid
from subprocess import Popen
import numpy as np
from ..iosupport import iosupport as io

matlabpath = matlab_path()

if matlabpath is None:
    raise ValueError('Cannot find matlab binary, please check installation')

def _write_script(script, tmpdir='tmp'):
    if check_write_access(tmpdir) is True: 
        u = uuid.uuid4()
        script_file = os.path.join(tmpdir, u, '.m')
        with open(script_file, 'w') as file:
            file.write(script)
        return script_file
    
def run_matlab(*scripts):
    """
    Run matlab scripts in sequence, accepts any number of scripts
    """
    options = ['-nosplash', '-nodesktop', '-r']
    command = []
    for script in scripts:
        command.append("run('{0}');".format(script))
    
    command.append("exit;")
    
    p = Popen(matlabpath + options + command)
    stderr, stdout = p.communicate()
    
    if stderr is not None:
        raise RuntimeError(stderr)
        print('Could run matlab, check stderr')
    
    # remove temp scripts
    for script in scripts:
        os.system(f'rm -f {script}')    

def wishart(obj, mask, tmpdir='tmp'):
    """
    Runs wishart filter from icaDim.m in HCPpipelines.

    Parameters
    ----------
    obj : str or ndarray
        Filename of input fmri file. Must have shape=(:,:,:,T) where T is the
        number of volumes.
    mask : str or ndarray
        Filename of input mask of fmri file, optional.
    tmpdir : str, optional
        Directory to place temporary script file. The default is '/tmp'.

    Returns
    -------
    ndarray of shape (:,:,:,T)
        Filtered fmri data

    """
    if isinstance(obj, np.ndarray) is True:
        # write out data to tempdir
        u = uuid.uuid4()
        fname = os.path.join(tmpdir, u, '.nii')
        io.write(obj, fname)
        obj = fname
    
    if not get_ext(obj) == '.nii':
        raise ValueError('Input file must have nifti extension')
        
    # write matlab script
    WISHART=f"""

    addpath functions/wishart
    addpath functions

    DEMDT = 1;
    VN = 1;
    Iterate = 2;
    NDist = 1;

    [~, data] = read('{obj}');

    [~, gm_msk] = read('{mask}');
    ind_gm = find(gm_msk);

    [folder, basename, ~] = fileparts('{obj}');

    T = size(data,4);
    x = zeros(T, length(ind_gm));

    for i=1:T
        tmp = data(:,:,:,i);
        x(i,:) = tmp(ind_gm);
    end

    Out = icaDim(x', DEMDT, VN, Iterate, NDist);

    x = Out.data';

    x = detrend(x, 'constant');
    x = x./repmat(std(x), T, 1);

    new_data = zeros(size(gm_msk,1), size(gm_msk,2), size(gm_msk,3), T);
    [xx, yy, zz] = ind2sub(size(gm_msk), ind_gm);

    x = x';

    for i = 1:length(ind_gm)
        new_data(xx(i), yy(i), zz(i), :) = x(i,:);
    end

    mat2nii(new_data, [folder, '/', basename, '_filtered.nii'], size(new_data), 32, '{mask}');
    """
    script = _write_script(WISHART, tmpdir)
    
    # run
    run_matlab(script)
    
    # now load filtered data
    data = io.read_nii()
    
    return data
    
