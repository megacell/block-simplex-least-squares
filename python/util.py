import ipdb

import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse.linalg as sla
import numpy as np
import numpy.linalg as la
from scipy.linalg import block_diag
import scipy.sparse as sps
import sys
import time
import scipy.io as sio
import logging

# Constraints
PROB_SIMPLEX = 'probability simplex'
# Reductions
EQ_CONSTR_ELIM = 'equality constraint elimination'
# Methods
L_BFGS = 'L-BFGS'
SPG = 'SPG'
ADMM = 'ADMM'

# Convenience functions
# -----------------------------------------------------------------------------

def is_sparse_matrix(A):
    return not sps.sputils.isdense(A)

def remove_zero_rows(M,x):
    nz = array(M.sum(axis=1).nonzero()[0])
    return M[nz,:], x[nz]

def remove_zero_cols(M,x):
    nz = array(M.sum(axis=0).nonzero()[1])
    return M[:,nz], x[nz]

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

def block_sizes_to_x0(block_sizes):
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

def x2z(x, block_sizes):
    """
    Convert x (original splits) to z variable (eliminated eq constraint)
    :param x:
    :param block_sizes:
    :return:
    """
    p = len(block_sizes)
    ind_end = np.cumsum(block_sizes)
    ind_start = np.hstack(([0],ind_end[:-1]))
    z = np.concatenate([np.cumsum(x[i:j-1]) for i,j \
                        in zip(ind_start,ind_end) if i<j-1])
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

def EQ_block_sort(EQ,x,M):
    """
    Reorder columns by blocks of flow, given by EQ, e.g. OD flow or waypoint flow

    :param EQ: incidence matrix for equality constraints
    :param A:
    :param x:
    :param M: optional matrix that also need columns sorted accordingly
    :return: rsort_index is the indices to reverse the sort
    """
    block_sizes = get_block_sizes(EQ)
    rank = EQ.nonzero()[0]
    sort_index = np.argsort(rank)
    M = col(M,sort_index)
    x = x[sort_index] # reorder
    EQ = col(EQ,sort_index)
    rsort_index = np.argsort(sort_index) # revert sort
    return (EQ.tocsr(),x,M.tocsr(),block_sizes,rsort_index)

def EQ_block_scale(EQ,EQx,x,M,m, thresh=1e-30):
    """
    Removes zero _blocks_ and scales matrices by block flow, given by EQ

    :param EQ: incidence matrix for equality constraints
    :param EQx: corresponding flow (EQ * x)
    :param A:
    :param x:
    :param M: optional matrix that also needs columns scaled/zeroed accordingly
    :param m: optional vector (M*x) that also need columns zeroed
    :return:
    """
    scaling =  EQ.T.dot(EQ.dot(x))
    nz = (scaling > thresh).nonzero()[0]
    x_split = np.nan_to_num(x / scaling)[nz]
    scaling = scaling[nz]
    DEQ = sps.diags([scaling],[0])
    M, m = remove_zero_rows(col(M,nz).dot(DEQ),m)
    EQ,EQx = remove_zero_rows(col(EQ,nz).dot(DEQ),EQx)
    assert la.norm(EQ.dot(x_split) - EQx) < 1e-10,\
        'Improper scaling: EQx != EQx, norm: %s' % la.norm(EQ.dot(x_split) - EQx)
    return (EQ.tocsr(),EQx,x_split,M.tocsr(),m,scaling)

def direct_solve(M,m,x_split=None):
    if M.shape[0] == M.shape[1]:
        if M.size == 1:
            x0 = array(m[0] / M[0,0]) if m[0] != 0 else 0
        else:
            x0 = sps.linalg.spsolve(M,m)
        if x_split is not None:
            error = np.linalg.norm(x0-x_split)
            logging.info('Exact solution, error: %s' % error)
    else:
        x0 = sps.linalg.lsmr(M,m)[0]
        if x_split is not None:
            error = np.linalg.norm(x0-x_split)
            logging.info('lsmr solution, error: %s' % error)
    return x0


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

def solver_input(data,full=False,L=True,OD=False,CP=False,LP=False,eq=None,
              init=False,thresh=1e-5,solve=False,damp=0.0,EQ_elim=True):
    """
    Load data from file about network state

    Notation:
    x_true = route flow
    x_split = route split

    :param filename:
    :param full: Use A_full, b_full instead of A,b
    :param OD: Extract information from T
    :param CP: Extract information from U
    :param eq: None uses block_sizes to generate equality constraint; OD uses
                T to generate equality constraint; CP uses U
    :return:
    """
    # Link-route and route
    # FIXME deprecate use of key 'x'
    output = {}

    # Load A,b if applicable
    A, b = None, None
    if L and full and 'A_full' in data and 'b_full' in data:
        A = sparse(data['A_full'])
        b = array(data['b_full'])
        if len(data['A'].shape) == 1:
            A = A.T
    elif L and 'A' in data and 'b' in data:
        A = sparse(data['A'])
        b = array(data['b'])
        if len(data['A'].shape) == 1:
            A = A.T
    elif 'phi' in data and 'b' in data:
        A = sparse(data['phi'])
        b = array(data['b'])
    if A is not None:
        assert_scaled_incidence(A)
    if 'b_full' in data:
        output['nAllLinks'] = array(data['b_full']).size
    if b is not None:
        output['nLinks'] = b.size

    # Load x_true
    if 'x_true' in data:
        x_true = array(data['x_true'])
        if len(x_true.shape) == 0:
            x_true = x_true.reshape((x_true.size))
    elif 'real_a' in data:
        x_true = array(data['real_a'])

    # Remove rows of zeros (unused sensors)
    if A is not None:
        nz = [i for i in xrange(A.shape[0]) if A[i,:].nnz == 0]
        nnz = [i for i in xrange(A.shape[0]) if A[i,:].nnz > 0]
        A, b = A[nnz,:], b[nnz]
        assert la.norm(A.dot(x_true) - b) < thresh, \
            'Check data input: Ax != b, norm: %s' % la.norm(A.dot(x_true) - b)
    AA,bb = A,b # Link constraints (NOTE: might still be None)

    n = x_true.shape[0]
    # OD-route
    if has_OD(data,OD):
        T,d = sparse(data['T']), array(data['d'])
        assert_simplex_incidence(T, n) # ASSERT
        output['nOD'] = d.size
        if solve:
            AA,bb = (T,d) if AA is None else (sps.vstack([AA,T]), np.append(bb,d))
    # Cellpath-route
    if has_CP(data,CP):
        U,f = sparse(data['U']), array(data['f'])
        assert_simplex_incidence(U, n) # ASSERT
        output['nCP'] = f.size
        if solve:
            AA,bb = (U,f) if AA is None else (sps.vstack([AA,U]), np.append(bb,f))
    # Linkpath-route + add to AA,bb
    if has_LP(data,LP):
        V,g = sparse(data['V']), array(data['g'])
        output['nLP'] = g.size
        AA,bb = (V,g) if AA is None else (sps.vstack([AA,V]), np.append(bb,g))
        logging.info('V: (%s,%s)' % (V.shape))

    if solve:
        if AA is None:
            output['error'] = "AA,bb is empty"
            return None,None,None,None,output

        from scipy.sparse.linalg import lsqr
        x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = \
            lsqr(AA,bb,damp=damp)
        output['istop'], output['init_iters'], output['r1norm'],output['r2norm'], \
        output['anorm'], output['acond'], output['arnorm'],output['xnorm'] = \
            istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm
        return AA, bb, x, x_true, output

    # Process equality constraints: scale by block, remove zero blocks, reorder
    block_sizes, rsort_index = None, None
    if eq == 'OD' and has_OD(data,OD):
        if has_CP(data,CP):
            AA,bb = (U,f) if AA is None else (sps.vstack([AA,U]), np.append(bb,f))
            logging.info('T: %s, U: %s' % (T.shape, U.shape))
        else:
            logging.info('T: (%s,%s)' % (T.shape))
        T,d,x_split,AA,bb,scaling = EQ_block_scale(T,d,x_true,AA,bb)
        T,x_split,AA,block_sizes,rsort_index = EQ_block_sort(T,x_split,AA)
        assert la.norm(T.dot(x_split) - d) < thresh, \
            'Check eq constraint Tx != d, norm: %s' % la.norm(T.dot(x_split)-d)
    elif eq == 'CP' and has_CP(data,CP):
        if has_OD(data,OD):
            AA,bb = (T,d) if AA is None else (sps.vstack([AA,T]), np.append(bb,d))
            logging.info('T: %s, U: %s' % (T.shape, U.shape))
        else:
            logging.info('U: (%s,%s)' % (U.shape))
        U,f,x_split,AA,bb,scaling = EQ_block_scale(U,f,x_true,AA,bb)
        U,x_split,AA,block_sizes,rsort_index = EQ_block_sort(U,x_split,AA)
        assert la.norm(U.dot(x_split) - f) < thresh, \
            'Check eq constraint Ux != f, norm: %s' % la.norm(U.dot(x_split)-f)
    else: # assume already sorted by blocks
        logging.warning('Use of deprecated clause')
        # TODO DEPRECATE
        x_split = x_true
        # TODO what is going on here????
        scaling = array(A.sum(axis=0)/(A > 0).sum(axis=0))
        scaling[np.isnan(scaling)]=0 # FIXME this is not accurate
        AA,bb = A,b
    assert la.norm(AA.dot(x_split) - bb) < thresh, \
        'Improper scaling: AAx != bb, norm: %s' % la.norm(AA.dot(x_split) - bb)

    if EQ_elim == False:
        if eq == 'OD' and has_OD(data,OD):
            return AA,bb,T,x_split,scaling,output
        elif eq == 'CP' and has_CP(data,CP):
            return AA,bb,U,x_split,scaling,output
        else:
            print 'Error: no eq constraint'
            return AA,bb,None,x_split,scaling,output

    logging.debug('Creating sparse N matrix')
    if block_sizes is not None:
        N = block_sizes_to_N(block_sizes)
    else:
        # In the case where there is no equality constraint, simply solve the
        # objective via iterative method
        N = None
        x0 = sps.linalg.lsmr(AA,bb)[0]
        return (AA, bb, N, block_sizes, x_split, nz, scaling, rsort_index, x0, output)

    logging.info('AA : %s, A : %s, blocks: %s' % (AA.shape, A.shape,
                                                  block_sizes.shape))

    logging.debug('File loaded successfully')
    if init:
        if eq == 'OD' and has_OD(data,OD):
            x0 = direct_solve(T,d,x_split=x_split)
        elif eq == 'CP' and has_CP(data,CP):
            x0 = direct_solve(U,f,x_split=x_split)
        else:
            x0 = np.array(block_e(block_sizes - 1, block_sizes))
    else:
        x0 = np.array(block_e(block_sizes - 1, block_sizes))

    return (AA, bb, N, block_sizes, x_split, nz, scaling, rsort_index, x0, output)

def load_data(filename,full=False,L=True,OD=False,CP=False,LP=False,eq=None,
              init=False,thresh=1e-5):
    """
    Load data from file about network state

    Notation:
    x_true = route flow
    x_split = route split

    :param filename:
    :param full: Use A_full, b_full instead of A,b
    :param OD: Extract information from T
    :param CP: Extract information from U
    :param eq: None uses block_sizes to generate equality constraint; OD uses
                T to generate equality constraint; CP uses U
    :return:
    """
    logging.debug('Loading %s...' % filename)
    data = sio.loadmat(filename)
    logging.debug('Unpacking...')

    AA, bb, N, block_sizes, x_split, nz, scaling, rsort_index, x0, output = \
        solver_input(data,full=full,L=L,OD=OD,CP=CP,LP=LP,eq=eq,init=init,
                     thresh=thresh)
    return AA, bb, N, block_sizes, x_split, nz, scaling, rsort_index, x0, output

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

# Assert functions
# -----------------------------------------------------------------------------

def assert_simplex_incidence(M,n):
    """
    1. Check that the width of the matrix is correct.
    2. Check that each column sums to 1
    3. Check that there are exactly n nonzero values
    :param M:
    :param n:
    :return:
    """
    assert M.shape[1] == n, 'Incidence matrix: wrong size'
    assert (M.sum(axis=0)-1).any() == False, \
        'Incidence matrix: columns should sum to 1'
    assert M.nnz == n, 'Incidence matrix: should be n nonzero values'

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
