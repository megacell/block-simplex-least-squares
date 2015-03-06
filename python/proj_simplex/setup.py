'''
Created on 5 nov. 2013

compile with command line 
python setup.py build_ext --inplace

@author: jerome thai
'''

import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

sourcefiles = ["proj_simplex_c.pyx"]
name = "proj_simplex_c"

setup(
    cmdclass = {"build_ext" : build_ext},
    ext_modules = [Extension(name,
            sourcefiles,
            include_dirs = [numpy.get_include()],
                        language = 'c++',
            )]
)
