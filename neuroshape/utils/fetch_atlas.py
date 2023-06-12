"""
Template helper function (for downloading from the internet, if necessary)

Based on neuromaps.datasets.atlases with an added helper for fs_LR_32k pial surface
"""

from collections import namedtuple
import os
from pathlib import Path

from nilearn.datasets.utils import _fetch_files
from sklearn.utils import Bunch

from neuromaps.datasets.utils import get_data_dir, get_dataset_info
from neuromaps.datasets.atlases import fetch_atlas

SURFACE = namedtuple('Surface', ('L', 'R'))
ALIAS = dict(
    fslr='fsLR', fsavg='fsaverage', mni152='MNI152', mni='MNI152',
    FSLR='fsLR', CIVET='civet'
)
DENSITIES = dict(
    civet=['41k', '164k'],
    fsaverage=['3k', '10k', '41k', '164k'],
    fsLR=['4k', '8k', '32k', '164k'],
    MNI152=['1mm', '2mm', '3mm'],
)

