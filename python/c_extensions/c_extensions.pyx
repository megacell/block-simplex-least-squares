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

def isotonic_regression_c(np.ndarray[double,ndim=1] y, start, end):
    cdef:
        double numerator
        int denominator
        Py_ssize_t i, pooled, n, k, e
    
    n = y.shape[0]
    assert start>=0 and start<n and end>0 and end<=n
    if start >= end: return y
    e = end-1
    while 1:
        # repeat until there are no more adjacent violators.
        i = start
        pooled = 0
        while i < e:
            k = i
            while k < e and y[k] >= y[k + 1]:
                k += 1
            if y[i] != y[k]:
                # y[i:k + 1] is a decreasing subsequence, so
                # replace each point in the subsequence with the
                # weighted average of the subsequence.
                numerator = 0.0
                for j in range(i, k + 1):
                    numerator += y[j]
                denominator = k + 1 - i
                for j in range(i, k + 1):
                    y[j] = numerator / denominator
                pooled = 1
            i = k + 1
        # Check for convergence
        if pooled == 0: break
 
    return y


def isotonic_regression_multi_c(np.ndarray[double,ndim=1] y, blocks):
    cdef:
        double numerator
        int denominator
        Py_ssize_t i, j, k, l, pooled, start, end, n, num_blocks
    n = y.shape[0]
    num_blocks = blocks.shape[0]
    assert False not in ((blocks[1:]-blocks[:-1])>0), 'block indices not increasing'
    assert blocks[0]>=0 and blocks[-1]<n, 'indices out of range'
    l = 0
    while l < num_blocks:
        start = blocks[l]
        end = n
        if l < num_blocks - 1: end = blocks[l+1]
        end -= 1
        while 1:
            # repeat until there are no more adjacent violators.
            i = start
            pooled = 0
            while i < end:
                k = i
                while k < end and y[k] >= y[k + 1]:
                    k += 1
                if y[i] != y[k]:
                    # y[i:k + 1] is a decreasing subsequence, so
                    # replace each point in the subsequence with the
                    # weighted average of the subsequence.
                    numerator = 0.0
                    for j in range(i, k + 1):
                        numerator += y[j]
                    denominator = k + 1 - i
                    for j in range(i, k + 1):
                        y[j] = numerator / denominator
                    pooled = 1
                i = k + 1
            # Check for convergence
            if pooled == 0: break

        l += 1


def isotonic_regression_c_2(np.ndarray[np.double_t,ndim=1] y, start, end):
    cdef:
        double numerator, previous
        int i, j, e, pooled, n, k, denominator
    n = y.shape[0]
    assert start>=0 and start<n and end>0 and end<=n
    if start >= end: return y
    cdef np.ndarray[np.int_t, ndim=1] weight = np.empty(n, dtype=np.int)
    i = 0
    while i < n:
        weight[i] = 1
        i += 1
    e = end
    while 1:
        # repeat until there are no more adjacent violators.
        i = start
        pooled = 0
        while i < e:
            k = i + weight[i]
            previous = y[i]
            while k < e and y[k] <= previous:
                previous = y[k]
                k += weight[k]
            if y[i] != previous:
                # y[i:k + 1] is a decreasing subsequence, so
                # replace each point in the subsequence with the
                # weighted average of the subsequence.
                numerator = 0.0
                denominator = 0
                j = i
                while j < k:
                    numerator += y[j] * weight[j]
                    denominator += weight[j]
                    j += weight[j]
                y[i] = numerator / denominator
                weight[i] = denominator
                pooled = 1
            i = k
        # Check for convergence
        if pooled == 0: break
     
    i = start
    while i < e:
        k = i + weight[i]
        j = i + 1
        while j < k:
            y[j] = y[i]
            j += 1
        i = k

    return y


def isotonic_regression_multi_c_2(np.ndarray[double,ndim=1] y, blocks):

    cdef:
        double numerator, previous
        int i, j, k, l, pooled, start, end, n, num_blocks, denominator
    n = y.shape[0]
    num_blocks = blocks.shape[0]
    assert False not in ((blocks[1:]-blocks[:-1])>0), 'block indices not increasing'
    assert blocks[0]>=0 and blocks[-1]<n, 'indices out of range'
    cdef np.ndarray[np.int_t, ndim=1] weight = np.empty(n, dtype=np.int)
    i = 0
    while i < n:
        weight[i] = 1
        i += 1
    l = 0
    while l < num_blocks:
        start = blocks[l]
        end = n
        if l < num_blocks-1: end = blocks[l+1]
        while 1:
            # repeat until there are no more adjacent violators.
            i = start
            pooled = 0
            while i < end:
                k = i + weight[i]
                previous = y[i]
                while k < end and y[k] <= previous:
                    previous = y[k]
                    k += weight[k]
                if y[i] != previous:
                    # y[i:k + 1] is a decreasing subsequence, so
                    # replace each point in the subsequence with the
                    # weighted average of the subsequence.
                    numerator = 0.0
                    denominator = 0
                    j = i
                    while j < k:
                        numerator += y[j] * weight[j]
                        denominator += weight[j]
                        j += weight[j]
                    y[i] = numerator / denominator
                    weight[i] = denominator
                    pooled = 1
                i = k
            # Check for convergence
            if pooled == 0: break
        l += 1

    i = 0
    while i < n:
        k = i + weight[i]
        j = i + 1
        while j < k:
            y[j] = y[i]
            j += 1
        i = k

    return y


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
             np.ndarray[np.double_t,ndim=1] Q,
             np.ndarray[np.double_t,ndim=1] c,
             np.ndarray[np.double_t,ndim=1] g):
             
    return quad_obj_c(<np.double_t*> x.data, 
                      <np.double_t*> Q.data, 
                      <np.double_t*> c.data, 
                      <np.double_t*> g.data, 
                      x.shape[0])


cdef quad_obj_c(np.double_t* x, np.double_t* Q, 
                np.double_t* c, np.double_t* g, Py_ssize_t n):
    cdef:
        np.double_t f
        Py_ssize_t i, j, k

    f = 0.0
    for i in range(n):
        g[i] = c[i]
        k = i*n
        for j in range(n): g[i] += Q[k+j] * x[j]
        f += .5 * (g[i] + c[i]) * x[i]
    return f

# x is current estimate
# f = f(x) objective at current estimate
# g = nabla_f(x) gradient at current estimate
# d is descent direction, d = x_new - x 
# x_new, f_new, g_new are for x_new = x + d
# performs backtracking line search between x and x_new
# see: http://stanford.edu/~boyd/cvxbook/, section 9.2

def line_search_quad_obj(np.ndarray[np.double_t,ndim=1] x,
                         np.double_t f,
                         np.ndarray[np.double_t,ndim=1] g,
                         np.ndarray[np.double_t,ndim=1] x_new,
                         np.double_t f_new,
                         np.ndarray[np.double_t,ndim=1] g_new,
                         np.ndarray[np.double_t,ndim=2] Q,
                         np.ndarray[np.double_t,ndim=1] c):
    cdef:
        np.double_t t, suffDec, upper_line, progTol, max
        Py_ssize_t i, j, n
    
    progTol = 1e-8
    n = x.shape[0]
    t = 1.
    suffDec = 1e-4

    # compute initial upper_line
    upper_line = f
    i = 0
    while i < n:
        upper_line += suffDec * g[i] * (x_new[i] - x[i])
        i += 1

    while f_new > upper_line:
        t *= .5

        # compute norm_inf of x - x_new
        i = 0
        max = 0
        while i < n:
            if x_new[i] - x[i] > max:
                max = x_new[i] - x[i]
            if x[i] - x_new[i] > max:
                max = x[i] - x_new[i]
            i += 1

        # Check whether step has become too small
        if t * max < progTol:
            t = 0.
            i = 0
            while i < n:
                x_new[i] = x[i]
                g_new[i] = g[i]
                i += 1
            f_new = f
            break
        
        # Compute new point
        i = 0
        while i < n:
            x_new[i] = x[i] + t * (x_new[i] - x[i])
            i += 1
        
        # compute objective and gradient at x_new
        i = 0
        f_new = 0
        while i < n:
            g_new[i] = c[i]
            j = 0
            while j < n:
                g_new[i] += Q[i,j] * x_new[j]
                j += 1
            f_new += .5 * (g_new[i] + c[i]) * x_new[i]
            i += 1

        # compute upper_line at x_new
        i = 0
        upper_line = f
        while i < n:
            upper_line += suffDec * g[i] * (x_new[i] - x[i])
            i += 1

    return x_new, f_new, g_new, t


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

