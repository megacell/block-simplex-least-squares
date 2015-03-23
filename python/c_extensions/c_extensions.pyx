'''
Created on 7 mar. 2015

C++11 wrapping with cython.
Compile with python setup.py build_ext --inplace

@author: jerome thai

'''

cimport numpy as np
import numpy as np
import ctypes


cdef extern from "proj_simplex.h":
    void proj_simplex(double *y, int start, int end)
    void proj_multi_simplex(double *y, int *blocks, int numblocks, int n)
    void proj_multi_ball(double *y, int *blocks, int numblocks, int n)


def proj_simplex_c(np.ndarray[np.double_t,ndim=1] y, start, end):
    n = y.shape[0]
    assert start>=0 and start<n and end>0 and end<=n
    if start >= end: return
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] y_c
    y_c = np.ascontiguousarray(y, dtype=np.double)
    proj_simplex(&y_c[0], start, end)


def proj_multi_simplex_c(np.ndarray[np.double_t,ndim=1] y, 
                     np.ndarray[np.int_t,ndim=1] blocks):
    assert False not in ((blocks[1:]-blocks[:-1])>0)
    assert blocks[0]>=0 and blocks[-1]<y.shape[0]
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] y_c
    cdef np.ndarray[int, ndim=1, mode="c"] b_c
    y_c = np.ascontiguousarray(y, dtype=np.double)
    b_c = np.ascontiguousarray(blocks, dtype=ctypes.c_int)
    proj_multi_simplex(&y_c[0], &b_c[0], blocks.shape[0], y.shape[0])


def proj_multi_ball_c(np.ndarray[np.double_t,ndim=1] y, 
                     np.ndarray[np.int_t,ndim=1] blocks):
    assert False not in ((blocks[1:]-blocks[:-1])>0)
    assert blocks[0]>=0 and blocks[-1]<y.shape[0]
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] y_c
    cdef np.ndarray[int, ndim=1, mode="c"] b_c
    y_c = np.ascontiguousarray(y, dtype=np.double)
    b_c = np.ascontiguousarray(blocks, dtype=ctypes.c_int)
    proj_multi_ball(&y_c[0], &b_c[0], blocks.shape[0], y.shape[0])


cdef extern from "isotonic_regression.h":
    void isotonic_regression(double *y, int start, int end)
    void isotonic_regression_multi(double *y, int *blocks, int numblocks, int n)
    void isotonic_regression_2(double *y, int start, int end)
    void isotonic_regression_multi_2(double *y, int *blocks, int numblocks, int n)
    void isotonic_regression_sparse(double *y, int start, int end, int *weight)


def isotonic_regression_c(np.ndarray[np.double_t,ndim=1] y, start, end):
    n = y.shape[0]
    assert start>=0 and start<n and end>0 and end<=n
    if start >= end: return
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] y_c
    y_c = np.ascontiguousarray(y, dtype=np.double)
    isotonic_regression(&y_c[0], start, end)


def isotonic_regression_multi_c(np.ndarray[np.double_t,ndim=1] y, 
                     np.ndarray[np.int_t,ndim=1] blocks):
    assert False not in ((blocks[1:]-blocks[:-1])>0)
    assert blocks[0]>=0 and blocks[-1]<y.shape[0]
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] y_c
    cdef np.ndarray[int, ndim=1, mode="c"] b_c
    y_c = np.ascontiguousarray(y, dtype=np.double)
    b_c = np.ascontiguousarray(blocks, dtype=ctypes.c_int)
    isotonic_regression_multi(&y_c[0], &b_c[0], blocks.shape[0], y.shape[0])


def isotonic_regression_c_2(np.ndarray[np.double_t,ndim=1] y, start, end):
    n = y.shape[0]
    assert start>=0 and start<n and end>0 and end<=n
    if start >= end: return
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] y_c
    y_c = np.ascontiguousarray(y, dtype=np.double)
    isotonic_regression_2(&y_c[0], start, end)


def isotonic_regression_multi_c_2(np.ndarray[np.double_t,ndim=1] y, 
                     np.ndarray[np.int_t,ndim=1] blocks):
    assert False not in ((blocks[1:]-blocks[:-1])>0)
    assert blocks[0]>=0 and blocks[-1]<y.shape[0]
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] y_c
    cdef np.ndarray[int, ndim=1, mode="c"] b_c
    y_c = np.ascontiguousarray(y, dtype=np.double)
    b_c = np.ascontiguousarray(blocks, dtype=ctypes.c_int)
    isotonic_regression_multi_2(&y_c[0], &b_c[0], blocks.shape[0], y.shape[0])


def isotonic_regression_sparse_c(np.ndarray[np.double_t,ndim=1] y, start, end,
                    np.ndarray[np.int_t,ndim=1] weight):
    n = y.shape[0]
    assert start>=0 and start<n and end>0 and end<=n
    if start >= end: return
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] y_c
    cdef np.ndarray[int, ndim=1, mode="c"] w_c
    y_c = np.ascontiguousarray(y, dtype=np.double)
    w_c = np.ascontiguousarray(weight, dtype=ctypes.c_int)
    isotonic_regression_sparse(&y_c[0], start, end, &w_c[0])


cdef extern from "quadratic_objective.h":
    double quad_obj(double *x, double *Q, double *c, double*g, int n)
    double line_search(double *x, double f, double *g, 
                       double *x_new, double f_new, double *g_new,
                       double *Q, double *c, int n)


def quad_obj_c(np.ndarray[np.double_t,ndim=1] x,
             np.ndarray[np.double_t,ndim=1] Q,
             np.ndarray[np.double_t,ndim=1] c,
             np.ndarray[np.double_t,ndim=1] g):
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] x_c
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] Q_c
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] c_c
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] g_c
    x_c = np.ascontiguousarray(x, dtype=np.double)
    Q_c = np.ascontiguousarray(Q, dtype=np.double)
    c_c = np.ascontiguousarray(c, dtype=np.double)
    g_c = np.ascontiguousarray(g, dtype=np.double)
    return quad_obj(&x_c[0], &Q_c[0], &c_c[0], &g_c[0], x.shape[0])


# x is current estimate
# f = f(x) objective at current estimate
# g = nabla_f(x) gradient at current estimate
# d is descent direction, d = x_new - x 
# x_new, f_new, g_new are for x_new = x + d
# performs backtracking line search between x and x_new
# see: http://stanford.edu/~boyd/cvxbook/, section 9.2

def line_search_quad_obj_c(np.ndarray[np.double_t,ndim=1] x,
                         np.double_t f,
                         np.ndarray[np.double_t,ndim=1] g,
                         np.ndarray[np.double_t,ndim=1] x_new,
                         np.double_t f_new,
                         np.ndarray[np.double_t,ndim=1] g_new,
                         np.ndarray[np.double_t,ndim=1] Q,
                         np.ndarray[np.double_t,ndim=1] c):
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] x_c
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] g_c
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] x_new_c
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] g_new_c
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] Q_c
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] c_c
    x_c = np.ascontiguousarray(x, dtype=np.double)
    g_c = np.ascontiguousarray(g, dtype=np.double)
    x_new_c = np.ascontiguousarray(x_new, dtype=np.double)
    g_new_c = np.ascontiguousarray(g_new, dtype=np.double)
    Q_c = np.ascontiguousarray(Q, dtype=np.double)
    c_c = np.ascontiguousarray(c, dtype=np.double)
    return line_search(&x_c[0], f, &g_c[0], &x_new_c[0], f_new, &g_new_c[0],
                       &Q_c[0], &c_c[0], x.shape[0])


def x2z_c(np.ndarray[np.double_t,ndim=1] x,
           np.ndarray[np.double_t,ndim=1] z,
           blocks):
    cdef:
        int i, j, k, n, start, end, num_blocks
        double tmp
    n = x.shape[0]
    assert False not in ((blocks[1:]-blocks[:-1])>0)
    assert blocks[0] == 0 and blocks[-1]<n
    k = 0
    j = 0
    num_blocks = blocks.shape[0]

    while k < num_blocks:
        start = blocks[k]
        end = n
        if k < num_blocks-1: end = blocks[k+1]
        i = start
        tmp = 0.0
        while i < end - 1:
            tmp += x[i]
            z[j] = tmp
            j += 1
            i += 1
        k += 1
    return z


def z2x_c(np.ndarray[np.double_t,ndim=1] x,
           np.ndarray[np.double_t,ndim=1] z,
           blocks):
    cdef:
        int i, j, k, n, start, end, num_blocks
        double tmp
    n = x.shape[0]
    assert False not in ((blocks[1:]-blocks[:-1])>0)
    assert blocks[0] == 0 and blocks[-1]<n
    k = 0
    j = 0
    num_blocks = blocks.shape[0]
    while k < num_blocks:
        start = blocks[k]
        end = n
        if k < num_blocks-1: end = blocks[k+1]
        i = start
        tmp = 0.0
        while i < end-1:
            x[i] = z[j] - tmp
            tmp = z[j]
            i += 1
            j += 1
        x[end-1] = 1.0 - tmp
        k += 1
    return x

