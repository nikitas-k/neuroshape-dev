.. image:: neuroshape_logo.png

The ``neuroshape`` toolbox is designed to run both functional and geometric eigenmodes using the Connectopic Laplacian approach for functional gradients `Haak et al. (2018) <https://www.sciencedirect.com/science/article/pii/S1053811917305463>`_, and the Laplace-Beltrami Operator on a finite vertex mesh as built in `ShapeDNA <https://github.com/Deep-MI/LaPy/tree/main>`_ (see also `Reuter et al. (2006) <http://dx.doi.org/10.1016/j.cad.2005.10.011>`_ and `Wachinger et al. (2015) <http://dx.doi.org/10.1016/j.neuroimage.2015.01.032>`_).

Installation requirements
-------------------------

``neuroshape`` works with Python 3.7+ and utilizes the following dependencies:

- nibabel (>=3.0)
- nilearn (>=0.7)
- numpy (>=1.14)
- scikit-learn (>=0.17)
- scipy
- lapy (>=0.7)
- scikit-sparse (>=0.4.8)
- neuromaps

**VERY IMPORTANT:**

In order to use much of the functionality of this code, including ``volume_eigenmodes.py`` and ``run_volume_eigenmodes_batch.sh``, you must:

1. Install `FreeSurfer <https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall>`_ and source it on your OS path.
2. Install `Connectome Workbench <https://www.humanconnectome.org/software/get-connectome-workbench>`_ and source it on your OS path.
3. Install `Gmsh <https://gmsh.info/>`_ and source it on your OS path.
4. Source MATLAB on your OS path.

`See instructions here on how to source binaries to path. <https://superuser.com/questions/284342/what-are-path-and-other-environment-variables-and-how-can-i-set-or-use-them>`_

The ``python`` script ``volume_eigenmodes.py`` was sourced from the `BrainEigenmodes <https://github.com/NSBLab/BrainEigenmodes/tree/main>`_ repository. Please cite their `Nature paper (Pang et al. 2023) <https://www.nature.com/articles/s41586-023-06098-1>`_ if you use that.

The MATLAB scripts in ``neuroshape/functions/wishart`` were sourced from the `HCPpipelines repository <https://github.com/Washington-University/HCPpipelines/tree/master/global/matlab/icaDim>`_ and related `Neuroimage paper (Glasser et al. 2013) <https://pubmed.ncbi.nlm.nih.gov/23668970/>`_. Please be sure to cite them if you use the ``--filter`` functionality in ``connectopic_laplacian.py``.

Installation
------------

Download from source:

.. code-block:: bash
  
  git clone https://github.com/breakspear/neuroshape

Additionally, as several C extensions must be built from source to use, install them with:

.. code-block:: bash

  git clone https://github.com/breakspear/neuroshape
  cd neuroshape
  python setup.py build
  python setup.py install

This will install the module in your environment's (or ``/usr/local/python/``) site-packages directory. You can then import the extension into your own code:

.. code-block:: python

  from neuroshape.eta import eta_squared
  similarity = eta_squared(matrix_2d)

.. _installation:

Citation
--------

If you use the ``neuroshape`` toolbox, please cite our paper .....
If you use the subroutines involved, such as ``lapy`` or ``volume_eigenmodes.py``, please be sure to cite the original authors.

License
-------

This work is licensed under a BSD 3-Clause "New" or "Revised" License.

Copyright 2023 Systems Neuroscience Lab

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors 
may be used to endorse or promote products derived from this software without 
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN 
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
