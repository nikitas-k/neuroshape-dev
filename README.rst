.. image:: https://raw.githubusercontent.com/nikitas-k/neuroshape-dev/main/docs/_static/neuroshape_logo.png?sanitize=true

The ``neuroshape`` toolbox is designed to run both functional gradients and geometric eigenmodes using the Connectopic Laplacian approach for resting-state functional gradients `Haak et al. (2018) <https://www.sciencedirect.com/science/article/pii/S1053811917305463>`_, our lab's approach (`Borne et al. (2023) <https://www.sciencedirect.com/science/article/pii/S1053811923001428>`_) to producing and analyzing task-driven gradients using psychophysiological interactions, and the Laplace-Beltrami Operator on a finite vertex mesh as built in `ShapeDNA <https://github.com/Deep-MI/LaPy/tree/main>`_ (see also `Reuter et al. (2006) <http://dx.doi.org/10.1016/j.cad.2005.10.011>`_ and `Wachinger et al. (2015) <http://dx.doi.org/10.1016/j.neuroimage.2015.01.032>`_).

Installation requirements
-------------------------

``neuroshape`` works with Python 3.8+ and utilizes the following dependencies:

- nibabel (>=3.0)
- nilearn (>=0.7)
- numpy (>=1.14)
- scikit-learn (>=0.17)
- scipy
- lapy (>=0.7)
- scikit-sparse (>=0.4.8)
- neuromaps

**VERY IMPORTANT:**

In order to use much of the functionality of this code, you must:

1. Install `FreeSurfer <https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall>`_ and source it on your OS path.
2. Install `Connectome Workbench <https://www.humanconnectome.org/software/get-connectome-workbench>`_ and source it on your OS path.
3. Install `Gmsh <https://gmsh.info/>`_ and source it on your OS path.
4. Source MATLAB on your OS path.
5. Install `scikit-sparse's libraries <https://github.com/scikit-sparse/scikit-sparse>`_. Follow the installation process over there first.

`See instructions here on how to source binaries to path. <https://superuser.com/questions/284342/what-are-path-and-other-environment-variables-and-how-can-i-set-or-use-them>`_

The ``python`` script ``volume_eigenmodes.py`` was sourced from the `BrainEigenmodes <https://github.com/NSBLab/BrainEigenmodes/tree/main>`_ repository. Please cite their `Nature paper (Pang et al. 2023) <https://www.nature.com/articles/s41586-023-06098-1>`_ if you use that.

The MATLAB scripts in ``neuroshape/functions/wishart`` were sourced from the `HCPpipelines repository <https://github.com/Washington-University/HCPpipelines/tree/master/global/matlab/icaDim>`_ and related `Neuroimage paper (Glasser et al. 2013) <https://pubmed.ncbi.nlm.nih.gov/23668970/>`_. Please be sure to cite them if you use the ``--filter`` functionality in ``connectopic_laplacian.py``.

Installation
------------

We have made it easy to download from source and install the dependencies automatically through a ``conda`` environment file (`install and initialize conda first <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_):

.. code-block:: bash
  
  git clone https://github.com/breakspear/neuroshape
  cd neuroshape
  conda env create -f environment.yml

To install scikit-sparse, however, as mentioned above, you will have to `follow their instructions first to install the libraries they depend on <https://github.com/scikit-sparse/scikit-sparse>`_. Come back here when you've done that, then type the above code into a terminal. This will "automatically" (tested and working on MacOS Big Sur v11.6 and MacOS Catalina 10.15.7) download and install the dependencies in an environment called ``neuroshape`` (unless this already exists somehow, in which case you will have to specify your own environment name with ``conda env create -f environment.yml -n myenvironment``).  After downloading the dependencies, several C extensions must be built from source to use:

.. code-block:: bash

  git clone https://github.com/breakspear/neuroshape
  cd neuroshape
  conda env create -f environment.yml
  python setup.py build
  python setup.py install

NOTE: The above must be performed in that order, otherwise the setup won't run properly. If you don't wish to initialize a whole new environment (or you don't use ``conda``), install the dependencies separately and forgo the ``conda env create`` step. Either way, the above will install the module in your environment's (or ``/usr/local/python/``) site-packages directory under the package ``neuroshape``. You can then import the extensions into your own code, e.g.:

.. code-block:: python

  from neuroshape.eta import eta_squared
  similarity = eta_squared(matrix_2d)

We are working on implementing full documentation for all extensions and tools in this package. As the project is in a rapid development stage, we appreciate your patience.

Citation
--------

If you use the ``neuroshape`` toolbox, please cite our paper .....
If you use the subroutines involved, such as ``lapy`` or ``volume_eigenmodes.py``, please be sure to cite the original authors.

License
-------

This work is licensed under a BSD 3-Clause "New" or "Revised" License.

Copyright (C) 2023 Systems Neuroscience Group Newcastle. Please read the full license `here <https://github.com/nikitas-k/neuroshape-dev/blob/main/LICENSE>`_ before use.
