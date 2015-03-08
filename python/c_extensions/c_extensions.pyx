'''
Created on 7 mar. 2015

C++11 wrapping with cython.
Compile with python setup.py build_ext --inplace

@author: jerome thai

'''

cimport numpy as np
import numpy as np

# PAV algorithm for isotonic regression adapted from:
# http://tullo.ch/articles/speeding-up-isotonic-regression/

def isotonic_regression_c(np.ndarray[np.double_t,ndim=1] y,
                         np.ndarray[np.double_t,ndim=1] weight, start, end):
    cdef:
        np.double_t numerator, denominator
        Py_ssize_t i, pooled, n, k
    
    n = y.shape[0]
    assert start>=0 and start<n and end>0 and end<=n
    if start >= end: return y
    while 1:
        # repeat until there are no more adjacent violators.
        i = start
        pooled = 0
        end -= 1
        while i < end:
            k = i
            while k < end and y[k] >= y[k + 1]:
                k += 1
            if y[i] != y[k]:
                # y[i:k + 1] is a decreasing subsequence, so
                # replace each point in the subsequence with the
                # weighted average of the subsequence.
                numerator = 0.0
                denominator = 0.0
                for j in range(i, k + 1):
                    numerator += y[j] * weight[j]
                    denominator += weight[j]
                for j in range(i, k + 1):
                    y[j] = numerator / denominator
                pooled = 1
            i = k + 1
        # Check for convergence
        if pooled == 0: break
 
    return y


def isotonic_regression_multi_c(np.ndarray[np.double_t,ndim=1] y,
                               np.ndarray[np.double_t,ndim=1] weight, blocks):
    n = y.shape[0]
    num_blocks = blocks.shape[0]
    assert False not in ((blocks[1:]-blocks[:-1])>0), 'block indices not increasing'
    assert blocks[0]>=0 and blocks[-1]<n, 'indices out of range'
    cdef:
        np.double_t numerator, denominator
        Py_ssize_t i, j, pooled, k, start, end
    j = 0
    while j < num_blocks:
        start = blocks[j]
        end = n
        if j < num_blocks-1: end = blocks[j+1]
        
        while 1:
            # repeat until there are no more adjacent violators.
            i = start
            pooled = 0
            end -= 1
            while i < end:
                k = i
                while k < end and y[k] >= y[k + 1]:
                    k += 1
                if y[i] != y[k]:
                    # y[i:k + 1] is a decreasing subsequence, so
                    # replace each point in the subsequence with the
                    # weighted average of the subsequence.
                    numerator = 0.0
                    denominator = 0.0
                    for j in range(i, k + 1):
                        numerator += y[j] * weight[j]
                        denominator += weight[j]
                    for j in range(i, k + 1):
                        y[j] = numerator / denominator
                    pooled = 1
                i = k + 1
            # Check for convergence
            if pooled == 0: break

        j += 1


cdef extern from "arrays.h":
    void proj_simplex(double *y, int start, int end)
    void proj_multi_simplex_hack(double *y, double *blocks, int numblocks, int n)

def proj_simplex_c(np.ndarray[np.double_t,ndim=1] y, start, end):
    n = y.shape[0]
    assert start>=0 and start<n and end>0 and end<=n
    if start >= end: return
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] y_c
    y_c = np.ascontiguousarray(y, dtype=np.double)
    proj_simplex(&y_c[0], start, end)


def proj_multi_simplex_c(np.ndarray[np.double_t,ndim=1] y, blocks):
    n = y.shape[0]
    assert False not in ((blocks[1:]-blocks[:-1])>0), 'block indices not increasing'
    assert blocks[0]>=0 and blocks[-1]<n, 'indices out of range'
    blocks = blocks.astype(np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] y_c
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] b_c
    y_c = np.ascontiguousarray(y, dtype=np.double)
    b_c = np.ascontiguousarray(blocks, dtype=np.double)
    proj_multi_simplex_hack(&y_c[0], &b_c[0], blocks.shape[0], y.shape[0])


def quad_obj(np.ndarray[np.double_t,ndim=1] x,
             np.ndarray[np.double_t,ndim=2] Q,
             np.ndarray[np.double_t,ndim=1] c,
             np.ndarray[np.double_t,ndim=1] g):
    cdef:
        np.double_t f
        Py_ssize_t i, j, n

    n = x.shape[0]
    i = 0
    f = 0
    while i < n:
        g[i] = c[i]
        j = 0
        while j < n:
            g[i] += Q[i,j] * x[j]
            j += 1
        f += .5 * (g[i] + c[i]) * x[i]
        i += 1
    return f, g
