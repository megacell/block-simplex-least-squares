import ipdb
import sys
import time
import logging
import warnings
import functools

import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse.linalg as sla
import numpy as np
import numpy.linalg as la
import scipy.linalg as ssla
import scipy.io
from scipy.linalg import block_diag
import scipy.sparse as sps
import scipy.io as sio

import ipdb

from algorithm_utils import quad_obj_np, normalization

# Constraints
PROB_SIMPLEX = 'probability simplex'
# Reductions
EQ_CONSTR_ELIM = 'equality constraint elimination'
# Methods
L_BFGS = 'L-BFGS'
SPG = 'SPG'
ADMM = 'ADMM'


def deprecated(func):
    '''This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.'''

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn_explicit(
            "Call to deprecated function {}.".format(func.__name__),
            category=Warning,
            filename=func.func_code.co_filename,
            lineno=func.func_code.co_firstlineno + 1
        )
        return func(*args, **kwargs)
    return new_func

# Convenience functions
# -----------------------------------------------------------------------------

def is_sparse_matrix(A):
    return not sps.sputils.isdense(A)

def remove_zero_rows(M,x):
    nz = array(M.sum(axis=1).nonzero()[0])
    return M[nz,:], x[nz], nz

def remove_zero_cols(M,x):
    nz = array(M.sum(axis=0).nonzero()[1])
    return M[:,nz], x[nz], nz

def row(M,m):
    return M.tocsr()[m,:].tocoo()

def col(M,m):
    return M.tocsc()[:,m].tocoo()

# Clean array wrapper
def array(x):
    return np.atleast_1d(np.squeeze(np.array(x)))

# Clean sparse matrix wrapper
def sparse(A):
    if A is None:
        return None
    if type(A) == np.ndarray:
        return sps.csr_matrix(A)
    return A.tocsr()

# Check if all entries equal
def all_equal(x,y):
    return np.all(np.equal(x,y))

# Check if all entries approximately equal
def almost_equal(x,y,tol=1e-8):
    return np.linalg.norm(x-y) < tol

# Check if ndarray consists of all ones
def is_ones(x):
    return np.all(x == np.ones(x.shape))

# Return a vector of size n with a 1 in the ith position
def e(i,n, val=1):
    x = np.zeros((n,1))
    x[i] = val
    return x

# Returns a vector with blocks of e, given a vector of sizes N and positions I
def block_e(I,N):
    return array(np.vstack([e(i,n) for (i,n) in zip(I,N)]))

def block_J(Js):
    return block_diag(Js)

def get_block_sizes(U):
    # Sum along rows
    return array((U>0).sum(axis=1)).astype(int)


def block_starts_to_block_sizes(block_starts, n):
    """! block_starts on containt the first index of each block
    n is the dimensionality of x
    """
    assert False not in ((block_starts[1:]-block_starts[:-1])>0)
    assert block_starts[0] == 0 and block_starts[-1] < n
    block_starts = np.array(block_starts)
    return np.append(block_starts[1:], [n]) - block_starts


def block_starts_to_x0(block_starts, n, f=None):
    """Convert block_starts to np.array vector x0 with 1 dimension
    """
    if f is None: f = np.ones(block_starts.shape[0])
    x0 = np.zeros(n)
    for i,j in zip(block_starts[1:]-1, f[:-1]): x0[i] = j
    x0[n-1] = f[-1]
    return x0


def block_sizes_to_x0_sparse(block_sizes):
    """Converts a list of the block sizes to a scipy.sparse vector x0
    """
    x0 = sps.dok_matrix((np.sum(block_sizes),1))
    for i in np.cumsum(block_sizes)-1: x0[(i,0)] = 1
    return x0.transpose()


def block_sizes_to_N(block_sizes):
    """Converts a list of the block sizes to a scipy.sparse matrix.

    The matrix will start in lil format, as this is the best way to generate it,
    but can easily be converted to another format such as csr for efficient multiplication.
    I will return it in csr so that each function doesn't need to convert it itself.
    """
    block_sizes = array(block_sizes)
    m = np.sum(block_sizes)
    n = m - block_sizes.shape[0]
    N = sps.lil_matrix((m, n))
    start_row = 0
    start_col = 0
    for i, block_size in enumerate(block_sizes):
        if block_size < 2:
            start_row += block_size
            start_col += block_size - 1
            continue
        for j in xrange(block_size-1):
            N[start_row+j, start_col+j] = 1
            N[start_row+j+1, start_col+j] = -1
        start_row += block_size
        start_col += block_size - 1
    return sparse(N)


def block_starts_to_N(block_starts, n, lasso=False):
    """Convert a list of the block_starts to numpy array N
    n is dimentionality of x

    if lasso: converts into lasso
    """
    block_sizes = np.append(block_starts[1:], [n]) - block_starts
    if lasso:
        N = np.zeros((n,n))
    else:
        N = np.zeros((n, n-block_sizes.shape[0]))
    start_row = 0
    start_col = 0
    for i, block_size in enumerate(block_sizes):
        if block_size >= 2:
            for j in xrange(block_size-1):
                N[start_row+j, start_col+j] = 1
                N[start_row+j+1, start_col+j] = -1
        if lasso:
            N[start_row+block_size-1, start_col+block_size-1] = 1
            start_col += 1
        start_row += block_size
        start_col += block_size - 1
    return N


def block_starts_to_M(block_starts, n, lasso=False):
    """Convert a list of the block_starts to numpy array M
    where M is the change of variable from z to x
    Be careful, if not lasso:
        n is the dimenionality of x
        block_starts is in the x variable
        converts z into x0, ..., x(n-1) for one block
    if lasso:
        x and z have same dimensionality
    """
    block_sizes = np.append(block_starts[1:], [n]) - block_starts
    if lasso:
        M = np.zeros((n,n))
    else:
        m = n - block_sizes.shape[0]
        M = np.zeros((m,m))
    ind = 0
    for i, block_size in enumerate(block_sizes):
        if lasso:
            end_block = block_size
        else:
            end_block = block_size-1
        for j in xrange(end_block):
            for k in xrange(j+1):
                M[ind+j, ind+k] = 1.0
        ind += end_block
    return M


def Q_to_Q_in_z(Q, block_starts, lasso=False):
    """Converts Q (Hessian of quadratic function)
    """
    n = Q.shape[0]
    N = block_starts_to_N(block_starts, n, lasso)
    return N.T.dot(Q).dot(N)


def construct_qp_from_least_squares(A, b):
    """0.5 * ||Ax-b||^2_2 = 0.5 x'Qx + c'x
    """
    Q = A.T.dot(A)
    c = -A.T.dot(b).flatten()
    return Q, c


def qp_to_qp_in_z(Q, c, block_starts, lasso=False, f=None):
    """Convert qp to qp in z 
    """
    n = Q.shape[0]
    if lasso:
        x0 = np.zeros(n)
    else:
        x0 = block_starts_to_x0(block_starts, n, f)
    N = block_starts_to_N(block_starts, n, lasso)
    Qz = Q_to_Q_in_z(Q, block_starts, lasso)
    cz = N.T.dot(c + Q.dot(x0))
    cz = cz.flatten()
    # f0 is the objective value at x=x0
    f0 = 0.5 * x0.T.dot(Q).dot(x0) + c.T.dot(x0)
    return Qz, cz, N, x0, f0


def ls_to_ls_in_z(A, b, block_starts, lasso=False, f=None):
    """Converts least squares to least squares in z
    """
    n = A.shape[1]
    if lasso:
        x0 = np.zeros(n)
    else:
        x0 = block_starts_to_x0(block_starts, n, f)   
    N = block_starts_to_N(block_starts, n, lasso)
    Az = A.dot(N)
    bz = b - A.dot(x0)
    return Az, bz, N, x0


def x2z(x, block_sizes=None, block_starts=None, lasso=False):
    """
    Convert x (original splits) to z variable (eliminated eq constraint)
    :param x:
    :param block_sizes:
    :return:
    """
    assert block_sizes is not None or block_starts is not None
    if block_sizes is None:
        n = x.shape[0]
        block_sizes = np.append(block_starts[1:], [n]) - block_starts
    p = len(block_sizes)
    ind_end = np.cumsum(block_sizes)
    ind_start = np.hstack(([0],ind_end[:-1]))
    if lasso:
        k = 0
    else:
        k = 1
    z = np.concatenate([np.cumsum(x[i:j-k]) for i,j \
                        in zip(ind_start,ind_end) if i<j-k])
    return z

# Helper functions
# -----------------------------------------------------------------------------

def load_weights(filename,block_sizes,weight=1):
    import pickle
    with open(filename) as f:
        data = pickle.load(f)
    D = np.array([v for (i,v) in data])
    # normalize weights
    blocks_end = np.cumsum(block_sizes)
    blocks_start = np.hstack((0,blocks_end[:-1]))
    blocks = [D[s:e] for s,e in np.vstack((blocks_start,blocks_end)).T]
    blocks = [b/sum(b) for b in blocks]
    return weight*np.array([e for b in blocks for e in b])


def load(filename, A=False, b=False, x_true=False):
    data = sio.loadmat(filename)
    if A:
        return sparse(data['A'])
    if b:
        return array(data['b'])
    if x_true:
        return array(data['x_true'])

def has_OD(data,OD):
    return OD and 'T' in data and 'd' in data and data['T'] is not None and \
           data['d'] is not None and data['T'].size > 0 and data['d'].size > 0

def has_CP(data,CP):
    return CP and 'U' in data and 'f' in data and data['U'] is not None and \
           data['f'] is not None and data['U'].size > 0 and data['f'].size > 0

def has_LP(data,LP):
    return LP and 'V' in data and 'g' in data and data['V'] is not None and \
           data['g'] is not None and data['V'].size > 0 and data['g'].size > 0


def particular_x0(block_sizes):
    return np.array(block_e(block_sizes - 1, block_sizes))

def AN(A,N):
    # TODO port from preADMM.m (lines 3-21)
    return A.dot(N)

def lsv_operator(A, N):
    """Computes largest singular value of AN

    Computation is done without computing AN or (AN)^T(AN)
    by using functions that act as these linear operators on a vector
    """

    # Build linear operator for AN
    def matmuldyad(v):
        return A.dot(N.dot(v))

    def rmatmuldyad(v):
        return N.T.dot(A.T.dot(v))
    normalized_lin_op = scipy.sparse.linalg.LinearOperator((A.shape[0],
                                                            N.shape[1]),
                                                           matmuldyad,
                                                           rmatmuldyad)

    # Given v, computes (N^TA^TAN)v
    def matvec_XH_X(v):
        return normalized_lin_op.rmatvec(normalized_lin_op.matvec(v))

    which='LM'
    v0=None
    maxiter=None
    return_singular_vectors=False

    # Builds linear operator object
    XH_X = scipy.sparse.linalg.LinearOperator(matvec=matvec_XH_X, dtype=A.dtype,
                                              shape=(N.shape[1], N.shape[1]))
    # Computes eigenvalues of (N^TA^TAN), the largest of which is the LSV of AN
    eigvals = sla.eigs(XH_X, k=1, tol=0, maxiter=None, ncv=10, which=which,
                       v0=v0, return_eigenvectors=False)
    lsv = np.sqrt(eigvals)
    # Take largest one
    return lsv[0].real


def timer(func, number= 1):
    '''
    Output the average time
    '''
    total = 0
    output = None
    for _ in xrange(number):
        if sys.platform == "win32":
            t = time.clock
        else:
            t = time.time
        start = t()
        output = func()
        end = t()
        total += end - start

    return output, total / number



def init_xz(block_sizes, x_true):
    """Generate initial points
    1: random
    2: by importance (cheating-ish)
    3: 10^importance (cheating-ish)
    4: uniform
    """
    n = np.sum(block_sizes)
    x1 = np.random.random_sample((n, 1))
    ind_end = np.cumsum(block_sizes)
    ind_start = np.hstack(([0],ind_end[:-1]))
    x1 = np.divide(x1, \
                   np.concatenate([np.sum(x1[i:j])*np.ones((k,1)) for i, j, k \
                                   in zip(ind_start,ind_end,block_sizes)]))

    tmp = np.concatenate([np.argsort(np.argsort(x_true[i:j])) for i,j in \
                          zip(ind_start,ind_end)]) + 1
    x2 = np.divide(tmp, \
                   np.squeeze(np.concatenate([np.sum(tmp[i:j])*np.ones((k,1)) \
                                              for i,j,k in zip(ind_start,
                                                               ind_end,
                                                               block_sizes)])))
    tmp = np.power(10, tmp)
    x3 = np.divide(tmp, \
                   np.squeeze(np.concatenate([np.sum(tmp[i:j])*np.ones((k,1)) \
                                              for i,j,k in zip(ind_start,
                                                               ind_end,
                                                               block_sizes)])))
    x4 = np.concatenate([(1./k)*np.ones((k,1)) for k in block_sizes])

    z1 = x2z(x1, block_sizes)
    z2 = x2z(x2, block_sizes)
    z3 = x2z(x3, block_sizes)
    z4 = x2z(x4, block_sizes)

    return x1,x2,x3,x4,z1,z2,z3,z4

def mask(arr):
    k = len(arr)
    size = np.max([len(arr[i]) for i in range(k)])
    masked = np.ma.empty((size,k))
    masked.mask = True
    for i in range(k):
        masked[:len(arr[i]),i] = np.array(arr[i])
    return masked

def stackMV(X,x,Y,y):
    """
    Stack matrix vector pair X,x on top of matrix vector pair Y,y
    :param X:
    :param x:
    :param Y:
    :param y:
    :return:
    """
    if X is None:
        return Y, y
    elif Y is None:
        return X, x
    elif X is None and Y is None:
        return None, None
    else:
        return sps.vstack([X,Y]), np.append(x,y)

# Assert functions
# -----------------------------------------------------------------------------

def assert_partial_simplex_incidence(M,n):
    """
    1. Check that the width of the matrix is correct.
    2. Check that each column sums to 1
    3. Check that there are no negative values
    4. Check that there are exactly n nonzero values
    :param M:
    :param n:
    :return:
    """
    assert M.shape[1] == n, 'Partial incidence: wrong size'
    assert np.where(M.sum(axis=0)==0)[1].size + \
           np.where(M.sum(axis=0)==1)[1].size == n, \
        'Partial incidence: columns should sum to only 1s and 0s'
    assert np.where(M.sum(axis=0)<0)[1].size == 0, \
        'Partial incidence: no negative values'
    assert M.nnz <= n, 'Partial incidence: should have <=n nonzero values'

def assert_simplex_incidence(M,n):
    """
    1. Check that the width of the matrix is correct.
    2. Check that each column sums to 1
    3. Check that there are no negative values
    4. Check that there are exactly n nonzero values
    :param M:
    :param n:
    :return:
    """
    assert M.shape[1] == n, 'Incidence matrix: wrong size'
    assert (M.sum(axis=0)-1).any() == False, \
        'Incidence matrix: columns should sum to 1'
    assert np.where(M.sum(axis=0)<0)[1].size == 0, \
        'Incidence matrix: no negative values'
    assert M.nnz == n, 'Incidence matrix: should have n nonzero values'

def assert_scaled_incidence(M,thresh=1e-12):
    """
    Check that all column entries are either 0 or the same entry value

    :param M:
    :return:
    """
    m,n = M.shape
    col_sum = M.sum(axis=0)
    col_nz = (M > 0).sum(axis=0)
    entry_val = np.array([0 if M[:,i].nonzero()[0].size == 0 else \
                              M[M[:,i].nonzero()[0][0],i] for i in range(n)])
    assert (np.abs(array(col_sum) - array(col_nz) * entry_val) < thresh).all(), \
        'Not a proper scaled incidence matrix, check column entries'


def generate_small_qp():
    Q = 2 * np.array([[2, .5], [.5, 1]])
    c = np.array([1.0, 1.0])
    x_true = np.array([.25, .75])       
    w, v = np.linalg.eig(Q) # w[-1] is the smallest eigenvalue
    f_min = 1.875
    min_eig = w[-1]
    return Q, c, x_true, f_min, min_eig


def random_least_squares(m, n, block_starts, sparsity=0.0, in_z=False,
                        lasso=False, truncated=False, distribution='normal'):
    """
    Generate least squares from the standard normal distribution
    m: # measurements
    n: # dimension of features
    block_starts: sizes of the blocks
    sparsity: sparsity in x_true
    in_z: if true, generate the normal distribution in z
    lasso: if true, generate x_true inside the l1-ball
    """
    assert sparsity < 1.0

    # construct A
    A = np.random.randn(m, n)
    if distribution == 'truncated':
        A = abs(A)
    if distribution == 'exponential':
        A = np.random.exponential(size=(m,n))
    if distribution == 'cumulative_normal':
        M = block_starts_to_M(block_starts, n, True)
        A = A.dot(M)
    if distribution == 'log_normal':
        A = np.random.lognormal(size=(m,n))
    if distribution == 'gamma':
        A = np.random.gamma(size=(m,n))

    # construct, sparsity, normalize x_true
    x_true = abs(np.random.randn(n,1))
    if int(sparsity * n) > 0:
        zeros = np.random.choice(n, sparsity * n, replace=False)
        for i in zeros: x_true[i] = 0.0
    block_ends = np.append(block_starts[1:], [n])
    normalization(x_true, block_starts, block_ends)

    # if LASSO, the projeciton is on the l1 ball
    if lasso:
        for start, end in zip(block_starts, block_ends):
            if np.random.uniform() > 0.7:
                alpha = np.random.uniform(0.5, 1)
                np.copyto(x_true[start:end], x_true[start:end] * alpha)
    # construct b, Q, c
    b = A.dot(x_true)
    x_true = x_true.flatten()
    Q, c = construct_qp_from_least_squares(A, b)
    w, v = np.linalg.eig(Q)
    f_min = quad_obj_np(x_true, Q, c)
    min_eig = w[-1]
    return {'Q':Q, 'c':c, 'x_true':x_true, 'f_min':f_min, 'min_eig':min_eig, 
            'A':A, 'b':b}


def coherence(A):
    """get coherence of A
    """ 
    A2 = np.copy(A)
    m, n = A2.shape
    for i in range(m):
        norm_row = np.linalg.norm(A2[i,:])
        if norm_row > 0.0: A2[i,:] = A2[i,:]/norm_row
    coherence = 0.0
    avg = 0.0
    for i in range(m):
        for j in range(i):
            product = abs(A2[i,:].dot(A2[j,:]))
            avg += product
            coherence = max(product, coherence)    
    return coherence, 2.*avg/(m*(m-1))


def generate_data(fname=None, n=100, m1=5, m2=10, A_sparse=0.5, alpha=1.0,
                  tolerance=1e-10, permute=False, scale=True, in_z=False,
                  distribution='uniform'):
    """
    A is m1 x n
    U is m2 x n

    :param fname: file to save to on disk
    :param n: size of x
    :param m1: number of measurements
    :param m2: number of blocks
    :param A_sparse: sparseness of A matrix
    :param alpha: prior for Dirichlet generating blocks of x
    :return:
    """
    if distribution == 'uniform':
        A = (np.random.random((m1, n)) > A_sparse).astype(np.float)
    if distribution == 'affine':
        tmp = 2 * (1 - A_sparse)
        line = (1 - tmp) + tmp *  np.arange(n)/(n-1)
        lines = []
        for i in range(m1):
            j = np.random.randint(n)
            lines.append(np.append(line[j:],line[:j]))
        A = (np.random.random((m1, n)) > np.array(lines)).astype(np.float)
    if distribution == 'aggregated':
        num_zeros = int(n*A_sparse)
        # in expectation, this doesn't give sparsity A_sparse, but close enough
        line = np.array([0.1]*num_zeros + [.9]*(n-num_zeros))
        lines = []
        for i in range(m1):
            j = np.random.randint(n)
            lines.append(np.append(line[j:],line[:j]))          
        A = (np.random.random((m1, n)) > np.array(lines)).astype(np.float)

    block_sizes = np.random.multinomial(n-m2,np.ones(m2)/m2) + np.ones(m2)
    assert sum(block_sizes) == n, 'all-zero row present!'
    block_starts = np.append([0], np.cumsum(block_sizes[:-1])).astype(int)
    # if in_z:
    #     M = block_starts_to_M(block_starts, n, True)
    #     A = A.dot(M)
    x = np.concatenate([np.random.dirichlet(alpha*np.ones(bs)) for bs in \
                        block_sizes])
    U = ssla.block_diag(*[np.ones(bs) for bs in block_sizes])
    if scale:
        f = np.floor(np.random.random(m2) * 1000)  # generate block scalings
        x = U.T.dot(f) * x  # scale x up by block
    else:
        f = np.ones(len(U))
    b = A.dot(x)  # generate measurements
    assert la.norm(U.dot(x)-f) < tolerance, "Ux!=f"
    assert la.norm(A.dot(x)-b) < tolerance, "Ax!=b"

    # permute the columns of A,U, entries of x
    if permute:
        reorder = np.random.permutation(n)
        A = A[:, reorder]
        U = U[:, reorder]
        x = x[reorder]
        assert la.norm(U.dot(x)-f) < tolerance, "Ux!=f after permuting"
        assert la.norm(A.dot(x)-b) < tolerance, "Ax!=b after permuting"

    data = { 'A': A, 'b': b, 'x_true': x, 'U': U, 'f': f, 'block_starts': block_starts, 'block_sizes': block_sizes}
    if fname:
        scipy.io.savemat(fname, data, oned_as='column')
    return data

if __name__ == "__main__":
    x = np.array([1/6.,2/6.,3/6.,1,.5,.1,.4])

    print "Demonstration of convenience functions (x2z, x2z)"
    block_sizes = np.array([3,1,3])
    z = x2z(x, block_sizes)
    x0 = block_sizes_to_x0(block_sizes)
    N = block_sizes_to_N(block_sizes)

    #print x
    #print z
    #print N.dot(z) +x0
    print init_xz(block_sizes, x)
