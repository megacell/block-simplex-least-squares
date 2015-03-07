'''
Created on 29 nov. 2014

Example of C++11 wrapping with cython.
Compile with python setup.py build_ext --inplace

@author: jerome thai

see
http://stackoverflow.com/questions/3046305/simple-wrapping-of-c-code-with-cython
'''

cimport numpy as np
import numpy as np


cdef extern from "arrays.h":
    void proj_simplex(double *y, int start, int end)
    void proj_multi_simplex_hack(double *y, double *blocks, int numblocks, int n)

def proj_simplex_c(np.ndarray[np.double_t,ndim=1] y, start, end):
    assert start>=0 and start<len(y) and end>0 and end<=len(y)
    if start >= end: return
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] y_c
    y_c = np.ascontiguousarray(y, dtype=np.double)
    proj_simplex(&y_c[0], start, end)

def proj_multi_simplex_c(np.ndarray[np.double_t,ndim=1] y, blocks):
    assert False not in ((blocks[1:]-blocks[:-1])>0), 'block indices not increasing'
    blocks = blocks.astype(np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] y_c
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] b_c
    y_c = np.ascontiguousarray(y, dtype=np.double)
    b_c = np.ascontiguousarray(blocks, dtype=np.double)
    proj_multi_simplex_hack(&y_c[0], &b_c[0], blocks.shape[0], y.shape[0])

