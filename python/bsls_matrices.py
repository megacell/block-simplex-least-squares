from __future__ import division
import logging

import scipy.sparse.linalg as sla
import numpy as np
import numpy.linalg as la
import scipy.sparse as sps
import scipy.io as sio

from bsls_utils import particular_x0, has_OD, has_CP, has_LP, block_sizes_to_N, \
    col, remove_zero_rows, get_block_sizes, array, sparse, \
    assert_partial_simplex_incidence, assert_scaled_incidence, \
    assert_simplex_incidence, stackMV

class BSLSMatrices:

    def __init__(self, data=None, fname=None, full=False, L=True, OD=False,
                 CP=False, LP=False, eq=None, init=False, thresh=1e-5,
                 noisy=False):
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
        """
        self.eq = eq

        # Load from data dict based on parameters (r indicates 'raw')
        if data is not None:
            self.rA, self.b, self.rx_true, self.rT, self.d, self.rU, self.f,\
            self.rV, self.g, self.nz, self.info = self.load_raw(data,
                full=full, L=L, OD=OD, CP=CP, LP=LP, thresh=thresh, noisy=noisy)
        elif fname is not None:
            self.rA, self.b, self.rx_true, self.rT, self.d, self.rU, self.f, \
            self.rV, self.g, self.nz, self.info = self.load_data(fname,
                full=full, L=L, OD=OD, CP=CP, LP=LP, thresh=thresh, noisy=noisy)

        self.A, self.T, self.U, self.V = self.rA, self.rT, self.rU, self.rV
        self.x_true, self.x_split = self.rx_true, self.rx_true
        self.block_sizes, self.rsort_index, self.scaling = None, None, None
        self.N, self.x0 = None, None

    def simple_simplex_form(self, thresh=1e-5, noisy=False):
        """
        1) Consolidates matrices into AA,bb
        2) Transforms into standard simplex form
        3) Cleans up zero rows
        4) Re-arranges so that C is in block-diagonal form
        :param thresh:
        :param noisy:
        :return:
        """
        self.consolidate(eq=self.eq)
        self.standard_simplex_form(thresh=thresh, noisy=noisy)
        self.cleanup()
        self.blockify(noisy=noisy)

        if self.AA is None or self.x_split is None:
            self.info['error'] = "AA,bb is empty"

    def degree_reduced_form(self, init=False):
        """
        In addition to transforming to simple simplex form, also produces
        nullspace basis N and initial feasible solution (in x)
        :param init:
        :return:
        """
        self.simple_simplex_form()
        logging.debug('Creating sparse N matrix')
        # Generate N if possible, otherwise, compute the lsmr solution
        if self.block_sizes is not None:
            self.N = block_sizes_to_N(self.block_sizes)
            self.x0 = self.initial_solution(init=init)
        else:
            # In the case where there is no equality constraint, simply solve the
            # objective via iterative method
            self.N = None
            self.x0 = sps.linalg.lsmr(self.AA,self.bb)[0]

    def consolidate(self, eq=None):
        """
        Consolidate matrices into AA, bb, C, d
        :return:
        """
        AA,bb = self.A, self.b
        AA,bb = stackMV(AA, bb, self.V, self.g)
        if eq == 'OD':
            self.AA, self.bb = stackMV(AA, bb, self.U, self.f)
            self.C, self.d = self.T, self.d  # may be None
            logging.info('U: %s' % repr(self.U.shape))
        elif eq == 'CP':
            self.AA, self.bb = stackMV(AA, bb, self.T, self.d)
            self.C, self.d = self.U, self.f  # may be None
        else:
            # Screw it, forget about constraints. Shove everything into AA,bb
            AA,bb = stackMV(AA, bb, self.T, self.d)
            AA,bb = stackMV(AA, bb, self.U, self.f)
            self.AA, self.bb = AA, bb
            self.C, self.d = None, None

    def blockify(self, noisy=False):
        """
        Re-arrange matrices so that C is blockwise diagonal
        :return:
        """
        self.block_sizes = get_block_sizes(self.C)
        rank = self.C.nonzero()[0]  # row "rank"
        # group the row "ranks" by selecting the corresponding column indicies
        sort_index = self.C.nonzero()[1][np.argsort(rank)]

        # re-arrange AA, x_true, x_split, and C
        self.AA = col(self.AA,sort_index).tocsr()
        self.x_true = self.x_true[sort_index] # reorder
        self.x_split = self.x_split[sort_index] # reorder
        self.C = col(self.C,sort_index).tocsr()

        # save the reverse sort index so we can undo the sort
        self.rsort_index = np.argsort(sort_index) # revert sort

    def cleanup(self):
        """
        Remove zero rows
        :return:
        """
        self.AA, self.bb, _ = remove_zero_rows(self.AA, self.bb)
        self.C, self.d, _ = remove_zero_rows(self.C, self.d)

    def standard_simplex_form(self, thresh=1e-30, noisy=False):
        """
        Transforms matrices into standard simplex form, so
            AA * x_true = bb --> tAA * x_split = bb
            C * x_true = d   --> C * x_split = 1
        :param thresh:
        :param noisy:
        :return:
        """
        # compute scaling for each entry of x
        scaling = self.C.T.dot(self.d)

        # remove the 'zero' blocks from x and C
        nz = (scaling > thresh).nonzero()[0]
        self.nz_cols = nz
        scaling = scaling[nz]
        self.x_split = np.nan_to_num(self.x_true[nz] / scaling)
        self.C = col(self.C, nz).tocsr()

        # remove the 'zero' blocks of AA; scale the rows of AA
        DEQ = sps.diags([scaling],[0])
        self.AA = col(self.AA, nz).dot(DEQ)

        # save the scaling
        self.scaling = scaling

    def initial_solution(self, init=False):
        if init and self.C is not None:
            ones = np.ones(self.d.shape)
            x0 = self.direct_solve(self.C, ones, x_split=self.x_split)
        else:
            x0 = particular_x0(self.block_sizes)
        return x0

    @staticmethod
    def reconstruct(x_split, rsort_index=None, scaling=None, nz=None, n=None):
        """
        Unsort, unzero, untransform
        :param x:
        :return:
        """
        x_unordered = x_split[rsort_index]  # un-order
        x_rescaled = x_unordered * scaling  # rescale
        x_true = np.zeros(n)
        x_true[nz] = x_rescaled  # unzero
        return x_true

    def load_data(self, filename, full=True, L=True, OD=True, CP=True, LP=True,
                  thresh=1e-5, noisy=False):
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

        A, b, x_true, T, d, U, f, V, g, nz, info = self.load_raw(data,
                    full=full,L=L,OD=OD,CP=CP,LP=LP, thresh=thresh,noisy=noisy)
        return A, b, x_true, T, d, U, f, V, g, nz, info

    def load_raw(self, data, full=False, L=True, OD=False, CP=False, LP=False,
                 thresh=1e-5, noisy=False, info=None):
        if info is None:
            info = {}

        # Load A,b if applicable
        A, b, nz = None, None, None
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
            info['nAllLinks'] = array(data['b_full']).size
        if b is not None:
            info['nLinks'] = b.size

        # Load x_true
        if 'x_true' in data:
            x_true = array(data['x_true'])
            if len(x_true.shape) == 0:
                x_true = x_true.reshape((x_true.size))
        elif 'real_a' in data:
            x_true = array(data['real_a'])
        else:
            return NotImplemented

        # Remove rows of zeros (unused sensors)
        if A is not None:
            nz = [i for i in xrange(A.shape[0]) if A[i,:].nnz == 0]
            nnz = [i for i in xrange(A.shape[0]) if A[i,:].nnz > 0]
            A, b = A[nnz,:], b[nnz]
            if not noisy:
                assert la.norm(A.dot(x_true) - b) < thresh, \
                    'Check data input: Ax != b, norm: %s' % la.norm(A.dot(x_true) - b)

        n = x_true.shape[0]

        T,d,U,f,V,g = None, None, None, None, None, None
        # OD-route
        if has_OD(data,OD):
            T,d = sparse(data['T']), array(data['d'])
            assert_partial_simplex_incidence(T, n) # ASSERT
            info['nOD'] = d.size
        # Cellpath-route
        if has_CP(data,CP):
            U,f = sparse(data['U']), array(data['f'])
            assert_simplex_incidence(U, n) # ASSERT
            info['nCP'] = f.size
        # Linkpath-route + add to AA,bb
        if has_LP(data,LP):
            V,g = sparse(data['V']), array(data['g'])
            info['nLP'] = g.size
            logging.info('V: (%s,%s)' % (V.shape))

        logging.info('A : (%s,%s)' % (A.shape if A is not None else (None,None)))
        return A, b, x_true, T, d, U, f, V, g, nz, info

    @staticmethod
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
            print 'Direct solve (M,m) =  (%s, %s)' % (repr(M.shape), repr(m.shape))
            x0 = sps.linalg.lsmr(M,m)[0]
            if x_split is not None:
                error = np.linalg.norm(x0-x_split)
                logging.info('lsmr solution, error: %s' % error)
        return x0

    ##########################################################################
    # Convenience access methods for different solvers
    ##########################################################################
    def get_LSQR(self):
        # FIXME triggered by 'solve' param
        self.consolidate(eq=None)
        return self.AA, self.bb, self.x_true, self.nz_cols

    def get_CS(self):
        return (self.AA, self.bb, self.N, self.block_sizes, self.x_split,
                self.nz_cols, self.scaling,
                self.rsort_index, self.x0, self.C)

    def get_BI(self):
        """
        Do not eliminate the equality constraint
        :return:
        """
        return self.AA, self.bb, self.C, self.x_split, self.scaling, \
               self.block_sizes

    def get_LS(self):
        logging.info('AA : %s, blocks: %s' % \
                     (self.AA.shape if self.AA is not None else None,
                      self.block_sizes.shape if self.block_sizes is not None else None))

        return (self.AA, self.bb, self.N, self.block_sizes, self.x_split,
                 self.nz_cols, self.scaling,
                self.rsort_index, self.x0)

