
import pickle
import numpy as np
import scipy.sparse as sps


__author__ = 'jeromethai'


def load_data_from_pkl(filepath):
    data = pickle.load(open( filepath, "rb" ))
    U = data['U']
    f = data['f']
    A = data['A']
    b = data['b']
    x_true = data['x_true']
    assert np.linalg.norm(A.dot(x_true)-b) < 1e-5
    assert np.linalg.norm(U.dot(x_true)-f) < 1e-5
    return A, b, U, f, x_true


def load_data_from_mat(filepath):
    data = scipy.io.loadmat(filepath)
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
    return np.array(block_sizes).astype(int)


def process_data(A, b, U, f, x_true):
    # permute U
    P = permute_column(U)
    U = U * P
    A = A * P
    x_true = P.T * x_true
    block_sizes = U_to_block_sizes(U)
    #A, b, U, f, x_true, block_sizes = remove_unused_paths(A, b, U, f, x_true, block_sizes)
    A, b, U, f, x_true, block_sizes = remove_size_one_blocks(A, b, U, f, x_true, block_sizes)
    A, b = remove_measurement(A, b)
    block_starts = np.append([0], np.cumsum(block_sizes)[:-1])
    assert np.linalg.norm(U.dot(x_true) - f) < 1e-5
    assert np.linalg.norm(A.dot(x_true) - b) < 1e-5
    return {'A': A, 'b': b, 'U': U, 'f': f, 'x_true': x_true, 
        'block_sizes': block_sizes, 'block_starts': block_starts}


def remove_size_one_blocks(A, b, U, f, x_true, block_sizes):
    row, col = [], []
    i, j = 0, 0
    for i, block_size in enumerate(block_sizes):
        if block_size == 1:
            row.append(i)
            col.append(j)
        j += block_size
    for j in col:
        b -= x_true[j] * np.squeeze(A[:,j])
    U = np.delete(U,row,0)
    U = np.delete(U,col,1)
    A = np.delete(A,col,1)
    f = np.delete(f,row,0)
    x_true = np.delete(x_true,col,0)
    assert np.linalg.norm(U.dot(x_true) - f) < 1e-5
    assert np.linalg.norm(A.dot(x_true) - b) < 1e-5
    return A, b, U, f, x_true, U_to_block_sizes(U)


def remove_unused_paths(A, b, U, f, x_true, block_sizes, tol=1e-12):
    ind = []
    for i in range(x_true.shape[0]):
        if x_true[i] < tol: ind.append(i)
    for j in ind:
        b -= x_true[j] * np.squeeze(A[:,j])
    A = np.delete(A,ind,1)
    U = np.delete(U,ind,1)
    x_true = np.delete(x_true,ind,0)
    assert np.linalg.norm(U.dot(x_true) - f) < 1e-5
    assert np.linalg.norm(A.dot(x_true) - b) < 1e-5
    return A, b, U, f, x_true, U_to_block_sizes(U)


def remove_measurement(A, b, thres=0.0):
    """Remove row if less than 'thres' routes go through the corresponding link
    """
    row = []
    for i in range(A.shape[0]): 
        if np.sum(A[i,:]) <= thres: row.append(i)
    A = np.delete(A,row,0)
    b = np.delete(b,row,0)
    return A, b


def remove_zeros_in_f(A, b, U, f, x_true, block_sizes, tol=1e-12):
    """remove zeros from f
    """
    row, col = [], []
    i, j = 0, 0
    for i, block_size in enumerate(block_sizes):
        if f[i] < tol:
            row.append(i)
            col += range(j, j+block_size)
        j += block_size
    for j in col:
        b -= x_true[j] * np.squeeze(A[:,j])
    U = np.delete(U,row,0)
    U = np.delete(U,col,1)
    A = np.delete(A,col,1)
    f = np.delete(f,row,0)
    x_true = np.delete(x_true,col,0)
    assert np.linalg.norm(U.dot(x_true) - f) < 1e-5
    assert np.linalg.norm(A.dot(x_true) - b) < 1e-5
    return A, b, U, f, x_true, U_to_block_sizes(U)


def load_and_process(filepath):
    """Load small network of L.A. and process it
    """
    A, b, U, f, x_true = load_data_from_pkl(filepath)
    return process_data(A, b, U, f, x_true)


def clean_progress(x, y):
    """Given a pandas dataframe with columns = ['time', 'f-f_min']
    clean data and return f-f_min in log10 scale
    also returns alpha, the rate of convergence
    """
    ind = [i for i in range(x.shape[0]) if y[i] <= 0.]
    x = np.delete(x,ind,0)
    log_y = np.log10(np.delete(y,ind,0))
    alpha = x.T.dot(log_y-log_y[0]) / x.T.dot(x)
    return x, log_y, alpha


def aggregate(A, x_true):
    """Shuffle the columns of A together to aggregate the ones together
    """
    pass


def aggregate_helper(A, x_true, start_col, end_col):
    """Recursively aggregate between start_col and end_col
    """
    pass


def row_with_most_ones(A, start_col, end_col):
    """Find the row with longest sequence of ones
    with length < end_col - start_col 
    """
    pass


if __name__ == '__main__':
    data = load_and_process('experiments/data/small_network_data.pkl')
    print data['f'].shape
    print data['U'].shape
    print data['x_true'].shape
    print data['x_true']
    print data['U'].dot(data['x_true'])
    print data['block_starts']
    print data['block_sizes']
    A = data['A']
    for i in range(A.shape[0]): print np.sum(A[i,:])
    print A.shape
    # A, b, U, f, x_true = load_data_from_mat('data/test_mat.mat')
    # print A.shape
    # print b.shape
    # print U.shape
    # print f.shape
    # print x_true.shape

