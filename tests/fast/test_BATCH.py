import unittest
import numpy as np
from cvxopt import matrix, spdiag, spmatrix, solvers
import sys
sys.path.append('../../')
from python.c_extensions.c_extensions import proj_simplex_c
from python.algorithm_utils import quad_obj_np, line_search_exact_quad_obj
import python.BATCH as batch
from python.bsls_utils import almost_equal

__author__ = 'jeromethai'

class TestBatch(unittest.TestCase):
  

    def setUp(self):
        seed = 237423433
        np.random.seed(seed)


    def test_batch(self):
        
        Q = 2 * np.array([[2, .5], [.5, 1]])
        c = np.array([1.0, 1.0])
        x_true = np.array([.25, .75])

        def proj(x):
            proj_simplex_c(x, 0, 2)

        def line_search(x, f, g, x_new, f_new, g_new, i):
            return line_search_exact_quad_obj(x, f, g, x_new, f_new, g_new, Q, c)

        def obj(x, g):
            return quad_obj_np(x, Q, c, g)

        x0 = np.array([.5, .5])
        f_min = 1.875

        sol = batch.solve(obj, proj, line_search, x0)
        assert almost_equal(sol['x'], x_true)
        assert sol['stop'][-10:] == '< prog_tol'
        sol = batch.solve(obj, proj, line_search, x0, f_min)
        assert almost_equal(sol['x'], x_true)
        assert sol['stop'][-10:] == ' < opt_tol'


    def test_batch_2(self):
        # test on constrainted least squares problem
        # min ||Ax-b||^2 = x'A'Ax - 2 b'Ax
        # s.t. ||x||_1 = 1, x>=0
        for i in range(5):
            m = 10 # number of measurements
            n = 7 # dimension of features
            A = np.random.randn(m, n)
            x_true = abs(np.random.randn(n,1))
            x_true = x_true / np.linalg.norm(x_true, 1)
            b = A.dot(x_true)
            Q = A.T.dot(A)
            c = -A.T.dot(b)
            c = c.flatten()
            x0 = np.ones(n) / n

            def proj(x):
                proj_simplex_c(x, 0, n)

            def line_search(x, f, g, x_new, f_new, g_new, i):
                return line_search_exact_quad_obj(x, f, g, x_new, f_new, g_new, Q, c)

            def obj(x, g):
                return quad_obj_np(x, Q, c, g)
                
            x_true = x_true.flatten()
            sol = batch.solve(obj, proj, line_search, x0, prog_tol=1e-14)
            assert almost_equal(sol['x'], x_true, 1e-5)


if __name__ == '__main__':
    unittest.main()