import unittest
import time
import numpy as np
import sys
sys.path.append('../../../')
from python.algorithm_utils import quad_obj_np, sparse_least_squares_obj
from python.c_extensions.c_extensions import quad_obj_c
from python.bsls_utils import generate_data, construct_qp_from_least_squares
import scipy.sparse as sps

__author__ = 'jeromethai'

class TestStressQuadObj(unittest.TestCase):

    def setUp(self):
        seed = 237423433
        np.random.seed(seed)
    

    def test_quad_obj_c(self):
        times_c = []
        times_np = []
        for n in [20, 200, 2000]:
            m = 1.5*n # number of measurements
            A = np.random.randn(m, n)
            x_true = abs(np.random.randn(n))
            x_true = x_true / np.linalg.norm(x_true, 1)
            b = A.dot(x_true)
            Q = A.T.dot(A)
            Q_flat = Q.flatten()
            c = -A.T.dot(b)
            c = c.flatten()
            g = np.zeros(n)

            def obj_c(x, g):
                return quad_obj_c(x, Q_flat, c, g)

            def obj_np(x, g):
                return quad_obj_np(x, Q, c, g)
            
            start_time = time.time()
            obj_c(x_true, g)
            times_c.append(time.time() - start_time)

            start_time = time.time()
            obj_np(x_true, g)
            times_np.append(time.time() - start_time)
        print 'times for quad_obj in Numpy', times_np
        print 'times for quad_obj in Cython', times_c


    def test_sparse_quad_obj(self):
        times_dense = []
        times_sparse = []
        for n in [20, 200, 2000]:
            m1 = n/4
            A_sparse = 0.9
            data = generate_data(n=n, m1=m1, A_sparse=A_sparse)
            A, b, x_true = data['A'], data['b'], data['x_true']
            A_sparse = sps.csr_matrix(A)
            A_sparse_T = sps.csr_matrix(A.T)
            Q, c = construct_qp_from_least_squares(A, b)
            Q_sparse = sps.csr_matrix(Q)

            def obj_np(x, g):
                return quad_obj_np(x, Q, c, g)

            def obj_sparse(x, g):
                return sparse_least_squares_obj(x, A_sparse_T, A_sparse, b, g)

            g = np.zeros(n)
            start_time = time.time()
            obj_np(x_true, g)
            times_dense.append(time.time() - start_time)
            start_time = time.time()
            obj_sparse(x_true, g)
            times_sparse.append(time.time() - start_time)
        print 'times for sparse QP', times_sparse
        print 'times for dense QP', times_dense
            # slow
            # start_time = time.time()
            # np.copyto(g, Q_sparse.dot(x_true) + c)
            # .5 * x_true.dot(g + c)
            # print 'time', time.time() - start_time
        # print g
        # print g2
        # Q, c = construct_qp_from_least_squares(data['A'], data['b'])
        # print Q
        # block_starts = data['block_starts'].astype(int)
        # Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
        # print Qz


if __name__ == '__main__':
    unittest.main()