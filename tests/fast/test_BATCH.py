import unittest
import numpy as np
from cvxopt import matrix, spdiag, spmatrix, solvers
import sys
sys.path.append('../../')
from python.c_extensions.c_extensions import (proj_simplex_c,
                                       quad_obj_c, 
                                       line_search_quad_obj_c)
import python.BATCH as batch
from python.bsls_utils import almost_equal

__author__ = 'jeromethai'

class TestBatch(unittest.TestCase):
  

    def setUp(self):
        seed = 237423433
        np.random.seed(seed)


    def test_batch(self):
        
        Q = 2 * np.array([[2, .5], [.5, 1]])
        Q_flat = Q.flatten()
        c = np.array([1.0, 1.0])
        x_true = np.array([.25, .75])

        def proj(x):
            proj_simplex_c(x, 0, 2)

        def line_search(x, f, g, x_new, f_new, g_new, i):
            return line_search_quad_obj_c(x, f, g, x_new, f_new, g_new, Q_flat, c)

        def obj(x, g):
            return quad_obj_c(x, Q_flat, c, g)

        x0 = np.array([.5, .5])
        f_min = 1.875

        sol = batch.solve(obj, proj, line_search, x0)
        assert almost_equal(sol['x'], x_true)
        assert sol['stop'] == 'stop with f_old-f < prog_tol'
        sol = batch.solve(obj, proj, line_search, x0, f_min)
        assert almost_equal(sol['x'], x_true)
        assert sol['stop'] == 'stop with f-f_min < opt_tol'


    def test_batch_2(self):
        # test on constrainted least squares problem
        # min ||Ax-b||^2 = x'A'Ax - 2 b'Ax
        # s.t. ||x||_1 = 1, x>=0
        for i in range(5):
            m = 10 # number of measurements
            n = 7 # dimension of features
            A = matrix(np.random.randn(m, n))
            x_true = abs(np.random.randn(n,1))
            x_true = x_true / np.linalg.norm(x_true, 1)
            b = A*matrix(x_true)
            Q = A.T * A
            c = -A.T*b
            # G = spdiag([-1.0]*n)
            # h = matrix([1.]*n, (n,1))
            # U = matrix([1.]*n, (1,n))
            # f = matrix(1.0)
            # sol = solvers.qp(Q, c, G, h, U, f)
            # assert almost_equal(sol['x'], x_true, 1e-6)

            Q = np.array(Q)
            Q_flat = Q.flatten()
            c = np.array(c).flatten()
            x0 = np.ones(n) / n

            def proj(x):
                proj_simplex_c(x, 0, n)

            def line_search(x, f, g, x_new, f_new, g_new, i):
                return line_search_quad_obj_c(x, f, g, x_new, f_new, g_new, Q_flat, c)

            def obj(x, g):
                return quad_obj_c(x, Q_flat, c, g)
                
            x_true = x_true.flatten()
            sol = batch.solve(obj, proj, line_search, x0, prog_tol=1e-14)
            assert almost_equal(sol['x'], x_true, 1e-5)


if __name__ == '__main__':
    unittest.main()