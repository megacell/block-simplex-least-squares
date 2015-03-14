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


if __name__ == '__main__':
    test_least_squares()