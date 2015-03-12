import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
from algorithm_utils import stopping


def least_squares(A, b, blocks, iters=1000, tolerance=1e-9):
    
    # initialize x
    n_vector = np.array(
            [([block_size]*block_size) for block_size in blocks])
    #n_vector = np.squeeze(n_vector.reshape((-1, 1)))
    #n_vector = n_vector.astype(float)
    n_vector = np.concatenate(n_vector).astype(float)
    x = np.divide(1.0, n_vector)
    x = np.squeeze(np.asarray(x))

    # mirror descent step
    if sps.issparse(A):
        Lf = sps.linalg.svds(A, 1, return_singular_vectors=False)[:]
        Lf = Lf[0]
    else:
        Lf = np.linalg.svd(A, compute_uv=False)
        Lf = Lf[0]

    def t_(k):
        return np.squeeze(np.asarray(
            np.sqrt(2*np.log(n_vector))/(np.sqrt(k)*Lf)))

    # compute objective and gradient
    def compute_gradient(x):
        inside = np.squeeze(np.asarray(A.dot(x))) - b
        return np.squeeze(
                np.asarray(A.T.dot(inside)))

    #main loop
    for _iter in xrange(1, iters+1):
        x_prev = x
        up = compute_gradient(x) 
        up *= t_(_iter)
        x = x * np.exp(-up)

        beginning = 0
        for block in blocks:
            x_section = x[beginning:block+beginning]
            x[beginning:block+beginning] = x_section/np.sum(x_section)
            beginning += block

        x = np.squeeze(np.asarray(x))
        if np.linalg.norm(x - x_prev, np.inf) < tolerance:
            break

    return x


def solve(obj, block_starts, x0, f_min=None, opt_tol=1e-6, 
          max_iter=5000, prog_tol=1e-9):
    """mirror descent algorithm
    """
    n = x0.shape[0]
    x = x0
    g = np.zeros(n)
    g_new = np.zeros(n)
    x_new = np.zeros(n)
    f_old = np.inf
    i = 1
    f = obj(x, g) # should update content of g
    block_ends = np.append(block_starts[1:], [n])
    while True:
        flag, stop = stopping(i, max_iter, f, f_old, opt_tol, prog_tol, f_min)
        if flag is True: break
        # update x
        np.copyto(x_new, x * np.exp(-g))
        #print x
        #print block_starts
        #print block_ends
        for start, end in zip(block_starts, block_ends):
            np.copyto(x_new[start:end], x_new[start:end] / np.sum(x_new[start:end]))
        f_new = obj(x_new, g_new)
        # take step
        f_old = f
        f = f_new
        np.copyto(x, x_new)
        np.copyto(g, g_new)
        i += 1
    return {'f': f, 'x': x, 'stop': stop, 'iterations': i}




if __name__ == '__main__':
    test_least_squares()