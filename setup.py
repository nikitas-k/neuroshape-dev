'''
    setup.py file for neuroshape module
'''

from setuptools import setup, Extension, find_packages
import numpy

# define the extension module
#eta = Extension('neuroshape.eta', sources=['src/eta_squared.c'],
#                          include_dirs=[numpy.get_include()])

#glmfit = Extension('neuroshape._stats', sources=['src/glmfit.c'],
#                   include_dirs=[numpy.get_include()])

#euler = Extension('neuroshape.euler', sources=['src/euler_threshold.c'],
#                  include_dirs=[numpy.get_include()])

# run the setup
setup(name='neuroshape',
      version='0.0.4.5',
      description="For computing connectopic and geometric Laplacian eigenmodes and performing null hypothesis testing. As implementation is ongoing, this description is subject to rapid change.",
      author='Nikitas C. Koussis, Systems Neuroscience Group Newcastle',
      author_email='nikitas.koussis@gmail.com',
      url='https://github.com/nikitas-k/neuroshape-dev',
      packages=find_packages())
      #ext_modules=[eta, glmfit, euler])
