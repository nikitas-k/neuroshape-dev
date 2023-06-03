# -*- coding: utf-8 -*-
"""
Utility functions
Taken from neuromaps/utils (c) netneurolab 2022
"""

import os

def check_fs_subjid(subject_id, subjects_dir=None):
    """
    Checks that `subject_id` exists in provided FreeSurfer `subjects_dir`

    Parameters
    ----------
    subject_id : str
        FreeSurfer subject ID
    subjects_dir : str, optional
        Path to FreeSurfer subject directory. If not set, will inherit from
        the environmental variable $SUBJECTS_DIR. Default: None

    Returns
    -------
    subject_id : str
        FreeSurfer subject ID, as provided
    subjects_dir : str
        Full filepath to `subjects_dir`

    Raises
    ------
    FileNotFoundError
    """

    # check inputs for subjects_dir and subject_id
    if subjects_dir is None or not os.path.isdir(subjects_dir):
        try:
            subjects_dir = os.environ['SUBJECTS_DIR']
        except KeyError:
            subjects_dir = os.getcwd()
    else:
        subjects_dir = os.path.abspath(subjects_dir)

    subjdir = os.path.join(subjects_dir, subject_id)
    if not os.path.isdir(subjdir):
        raise FileNotFoundError('Cannot find specified subject id {} in '
                                'provided subject directory {}.'
                                .format(subject_id, subjects_dir))

    return subject_id, subjects_dir