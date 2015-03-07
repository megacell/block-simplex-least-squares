'''
Created on 6 mar. 2015

compile with command line 
python setup.py build_ext --inplace

@author: jerome thai
'''

import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

sourcefiles = ["isotonic_regression_c.pyx"]
name = "isotonic_regression_c"

simplex_projection = Extension("simplex_projection",
        sources = ["simplex_projection.c"],
        include_dirs=[numpy.get_include()])

setup(
    cmdclass = {"build_ext" : build_ext},
    ext_modules = [Extension(name,
            sourcefiles,
            include_dirs = [numpy.get_include()],
                        language = 'c++',
            ), simplex_projection]
)
