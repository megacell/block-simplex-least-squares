

import numpy as np

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


def quad_obj_np(x, Q, c, g):
    """Receives numpy arrays
    """
    np.copyto(g, Q.dot(x) + c)
    f = .5 * x.T.dot(g + c)
    return f


def line_search_quad_obj_np(x, f, g, x_new, f_new, g_new, Q, c):
    """Backtracking line search
    """
    t = 1.0
    suffDec = 1e-4
    progTol = 1e-8
    upper_line = f + suffDec * g.dot(x_new - x)
    while f_new > upper_line:
        t *= .5
        # Check whether step has become too small
        if np.linalg.norm(x_new - x, np.inf)  < progTol:
            t = 0.0
            f_new = f
            np.copyto(g_new, g)
            np.copyto(x_new, x)
            break
        # update
        np.copyto(x_new, (1.0-t)*x + t*x_new)
        np.copyto(g_new, Q.dot(x_new) + c)
        f_new = .5 * x_new.T.dot(g_new + c)
        upper_line = f + suffDec * g.dot(x_new - x)

    return f_new


def line_search_exact_quad_obj(x, f, g, x_new, f_new, g_new, Q, c):
    """Exact line search
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
