from distutils.core import setup, Extension
import numpy

# define the extension module
simplex_projection = Extension("simplex_projection", sources = ["simplex_projection.c"],
                                  include_dirs=[numpy.get_include()])

setup(
   ext_modules = [
       simplex_projection,
   ],
)
