import pickle
import scipy.sparse as sps
#import pandas as pd
import numpy as np


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


def U_to_block_sizes(U):
    """Make sure U is properly set up!
    """
    block_sizes = []
    for i in range(U.shape[0]): block_sizes.append(np.sum(U[i,:]))
    return np.array(block_sizes)


def process_data(A, b, U, f, x_true):
    # permute U
    P = permute_column(U)
    U = U * P
    A = A * P
    x_true = P.T * x_true
    # block_sizes = U_to_block_sizes(U)
    # s = block_sizes.shape[0]
    # j = 0
    # # normalize the constraints to the simplex
    # for i in range(s):
    #     np.copyto(x_true[j:j+block_sizes[i]], x_true[j:j+block_sizes[i]] / f[i])
    #     j += block_sizes[i]
    # f = np.ones(s)
    # b = A.dot(x_true)
    assert np.linalg.norm(U.dot(x_true) - f) < 1e-5
    assert np.linalg.norm(A.dot(x_true) - b) < 1e-5
    return A, b, U, f, x_true


def load_and_process(filepath):
    A, b, U, f, x_true = load_data(filepath)
    return process_data(A, b, U, f, x_true)


if __name__ == '__main__':
    A, b, U, f, x_true = load_and_process('experiments/data/small_network_data.pkl')
    print f.shape
    print U.shape
    print x_true.shape
    print U.dot(x_true)

