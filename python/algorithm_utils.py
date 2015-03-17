
from python.c_extensions.c_extensions import (proj_multi_simplex_c,  
                                              isotonic_regression_multi_c)
import numpy as np
import scipy.sparse as sps
import pandas as pd

__author__ = 'jeromethai'


def proj_PAV(y, w=None):
    """PAV algorithm with box constraints
    """
    #y, w, l, u = s

    # if y.size != w.size:
    #     print y
    #     print w
    #     raise Exception("Shape of y (%s) != shape of w (%d)" % (y.size, w.size))

    n = len(y)
    if w is None: w = np.ones(n)
    y = y.astype(float)
    # x=y.copy()
    x=y

    if n==2:
        if y[0]>y[1]:
            x = (w.dot(y)/w.sum())*np.ones(2)
    elif n>2:
        j=range(n+1) # j contains the first index of each block
        ind = 0

        while ind < len(j)-2:
            if weighted_block_avg(y,w,j,ind+1) < weighted_block_avg(y,w,j,ind):
                j.pop(ind+1)
                while ind > 0 and weighted_block_avg(y,w,j,ind-1) > weighted_block_avg(y,w,j,ind):
                    if weighted_block_avg(y,w,j,ind) <= weighted_block_avg(y,w,j,ind-1):
                        j.pop(ind)
                        ind -= 1
            else:
                ind += 1

        for i in xrange(len(j)-1):
            x[j[i]:j[i+1]] = weighted_block_avg(y,w,j,i)*np.ones(j[i+1]-j[i])

    #return np.maximum(l,np.minimum(u,x))
    return x


# weighted average
def weighted_block_avg(y,w,j,ind):
    wB = w[j[ind]:j[ind+1]]
    return np.dot(wB,y[j[ind]:j[ind+1]])/wB.sum()


# see reference: http://arxiv.org/pdf/1309.1541.pdf
# vectorized numpy implementation

def proj_simplex(y, start, end):
    """projects subvector of y in range(start, end)"""
    assert start>=0 and start<len(y) and end>0 and end<=len(y)
    if start >= end: return
    x = np.sort(y[start:end])[::-1]
    tmp = np.divide((np.cumsum(x)-1), np.arange(1, end-start+1))
    y[start:end] = np.maximum(y[start:end] - tmp[np.sum(x>tmp)-1], 0)


def proj_multi_simplex(y, blocks):
    assert False not in ((blocks[1:]-blocks[:-1])>0), 'block indices not increasing'
    assert blocks[0]>=0 and blocks[-1]<len(y), 'indices out of range'
    for start, end in zip(blocks[:-1], blocks[1:]): proj_simplex(y, start, end)
    proj_simplex(y, blocks[-1], len(y))


def quad_obj_np(x, Q, c, g=None):
    """Receives numpy arrays
    """
    if g is None: g = np.zeros(x.shape[0])
    np.copyto(g, Q.dot(x) + c)
    f = .5 * x.T.dot(g + c)
    return f


def sparse_least_squares_obj(x, A_sparse_T, A_sparse, b, g):
    """Sparse computation of least squares objective and gradient
    """
    tmp = A_sparse.dot(x) - b
    np.copyto(g, A_sparse_T.dot(tmp))
    f = .5 * tmp.T.dot(tmp)
    return f


def decreasing_step_size(i, t0, alpha):
    """step size of the form t = t0 / (1 + t0*alpha*t)
    """
    progTol = 1e-8
    return t0 / (alpha*i + t0)
    # if np.linalg.norm(x_new - x, np.inf)  < progTol:
    #     f_new = f
    #     np.copyto(g_new, g)
    #     np.copyto(x_new, x)
    # t = t0 / (alpha*i + t0)
    # np.copyto(x_new, (1.0-t)*x + t*x_new)
    # np.copyto(g_new, Q.dot(x_new) + c)
    # f_new = .5 * x_new.T.dot(g_new + c)
    # return f_new


def line_search_np(x, f, g, x_new, f_new, g_new, obj):
    """Backtracking line search for quadratic objective
    """
    #print 'doing line_search'
    t = 1.0
    #suffDec = 1e-4
    suffDec = 1e-4
    progTol = 1e-12
    upper_line = f + suffDec * g.dot(x_new - x)
    while f_new > upper_line:
        #print 'Smaller step size'
        t *= .8
        # Check whether step has become too small
        step = np.linalg.norm(x_new - x, np.inf)
        if step  < progTol:
            #print 'Step {} too small in line search'.format(step)
            t = 0.0
            f_new = f
            np.copyto(g_new, g)
            np.copyto(x_new, x)
            break
        # update
        np.copyto(x_new, (1.0-t)*x + t*x_new)
        f_new = obj(x_new, g_new)
        upper_line = f + suffDec * g.dot(x_new - x)
    return f_new


def line_search_exact_quad_obj(x, f, g, x_new, f_new, g_new, Q, c):
    """Exact line search for quadratic objective
    """
    progTol = 1e-8
    d = x_new - x
    # Check whether step has become too small
    if np.linalg.norm(d, np.inf)  < progTol:
        t = 0.0
        f_new = f
        np.copyto(g_new, g)
        np.copyto(x_new, x)
        return f_new
    tmp = Q.dot(d)
    t = - (x.T.dot(tmp) + d.T.dot(c)) / d.T.dot(tmp)
    np.copyto(x_new, x + t*d)
    return quad_obj_np(x_new, Q, c, g_new) # returns f_new


def stopping(i, max_iter, f, f_old, opt_tol, prog_tol, f_min=None):
    """Simple stopping
    """
    flag = False
    stop = 'continue'
    if i == max_iter:
        stop = 'max_iter';
        flag = True
    if f_min is not None and f-f_min < opt_tol:
        stop = 'f-f_min = {} < opt_tol'.format(f-f_min)
        flag = True
    if abs(f_old-f) < prog_tol:
        stop = '|f_old-f| = {} < prog_tol'.format(abs(f_old-f))
        flag = True
    return flag, stop


def normalization(x, block_starts, block_ends):
    """Normalize x
    """
    for start, end in zip(block_starts, block_ends):
        np.copyto(x[start:end], x[start:end] / np.sum(x[start:end]))


def get_solver_parts(data, block_starts, min_eig, in_z=False, 
                    is_sparse=False, lasso=False):
    """Returns the step_size, proj, line_search, and obj functions
    for the least squares problem

    Parameters
    ----------
    data: data=(Q,c) if not sparse, data=(A, b) is sparse
    block_starts: first indices of each block
    min_eig: minimum eigenvalue of Q = A.T.dot(A)
    in_z: if variable expressed in z or not
    is_sparse: if we consider general QP of sparse least-squares
    """
    if is_sparse:
        A, b = data
        A_sparse = sps.csr_matrix(A)
        A_sparse_T = sps.csr_matrix(A.T)
        #n = A.shape[1]
        def obj(x, g=None):
            return sparse_least_squares_obj(x, A_sparse_T, A_sparse, b, g)
    else:
        Q, c = data
        #n = Q.shape[0]
        def obj(x, g=None):
            return quad_obj_np(x, Q, c, g)

    def step_size(i):
        return decreasing_step_size(i, 1.0, min_eig)

    if in_z:
        tmp = np.copy(block_starts)
        for i in range(len(tmp)):
            tmp[i] -= i
        def proj(x):
            isotonic_regression_multi_c(x, tmp)
            np.maximum(0.,x,x)
            np.minimum(1.,x,x)
    else:
        if lasso:
            def proj(x):
                proj_multi_ball_c(x, block_starts)
        else:
            def proj(x):
                proj_multi_simplex_c(x, block_starts)


    def line_search(x, f, g, x_new, f_new, g_new, i):
        return line_search_np(x, f, g, x_new, f_new, g_new, obj)

    return step_size, proj, line_search, obj


def save_progress(progress, f_min, name):
    """Save progress in a pandas database

    Parameters
    ----------
    progress: progress output by the solvers in BATCH.py
    f_min: minimum objective value
    name: name of the experiment
    """
    columns = ['time', 'f-f_min']
    iters = len(progress)
    index = pd.MultiIndex.from_tuples(zip([name]*iters, range(iters)))
    for i in range(len(progress)): progress[i][1] -= f_min
    df = pd.DataFrame(progress, index = index, columns = columns)
    return df
