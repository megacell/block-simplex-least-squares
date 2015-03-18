import pickle
import scipy.sparse as sps
#import pandas as pd
import numpy as np
from bsls_utils import (block_starts_to_N)

__author__ = 'jeromethai'


def load_data(filepath):
    data = pickle.load(open( filepath, "rb" ))
    U = data['U']
    f = data['f']
    A = data['A']
    b = data['b']
    x_true = data['x_true']
    assert np.linalg.norm(A.dot(x_true)-b) < 1e-5
    assert np.linalg.norm(U.dot(x_true)-f) < 1e-5
    return A, b, U, f, x_true


def check_rows(U):
    """Check rows of U to make sure that the ones are aggregated together
    """
    m, n = U.shape
    for i in range(m):
        switch = 0
        flag = 0
        for j in range(n):
            assert U[i,j] == 0 or U[i,j] == 1
            if switch == 0 and U[i,j] == 1.0:
                switch = 1
                flag += 1
            if switch == 1 and U[i,j] == 0.0:
                switch = 0
                flag += 1
            assert flag <= 2 # no more than one contiguous sequence of ones 
        assert flag >= 1 # at least one contiguous sequence of ones


def find_first_indices(U):
    """Find row, column of first one of each sequence as well as length
    """
    check_rows(U)
    indices = {} # {i-start: [j-start, length]}
    m, n = U.shape
    j = 0
    while j < n:
        i = 0
        while U[i,j] == 0.0: i += 1 # could be accelerated via bisection algorithm
        length = int(np.sum(U[i,:]))
        indices[i] = (j, length)
        j += length
    return indices


def permute_column(U):
    """Given an OD-path incidence matrix, generate a permutation matrix such that
    P.dot(U) permutes the columns of U such that U is block diagonal
    """
    check_rows(U)
    indices = find_first_indices(U)
    m, n = U.shape
    row = []
    col = []
    j_start = 0
    for i in range(m):
        i_start, length = indices[i]
        for i in range(length):
            row.append(i_start + i)
            col.append(j_start + i)
        j_start += length
    data = np.ones(n)
    return sps.csr_matrix((data, (np.array(row), np.array(col))), shape=(n, n))




if __name__ == '__main__':
    A, b, U, f, x_true = load_data('experiments/data/small_network_data.pkl')
    P = permute_column(U)
    print U * P
