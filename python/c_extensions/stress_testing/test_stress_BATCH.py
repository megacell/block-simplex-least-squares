import unittest
import time
import numpy as np
from cvxopt import matrix, spdiag, spmatrix, solvers
import sys
sys.path.append('../../../')
from python.c_extensions.c_extensions import (proj_simplex_c,
                                       quad_obj, 
                                       line_search_quad_obj)
import python.BATCH as batch
from python.bsls_utils import almost_equal

__author__ = 'jeromethai'

class TestStressBatch(unittest.TestCase):
  

    def setUp(self):
        seed = 237423433
        np.random.seed(seed)
    

    def test_batch(self):
        # test on constrainted least squares problem
        # min ||Ax-b||^2 = x'A'Ax - 2 b'Ax
        # s.t. ||x||_1 = 1, x>=0
        solvers.options['show_progress'] = False
        times_cvxopt = []
        times_batch = []
        iters_cvxopt = []
        iters_batch = []
        precision_cvxopt = []
        precision_batch = []
        for n in [10, 100, 300]: # dimension of features
            m = 1.5*n # number of measurements
            A = np.random.randn(m, n)
            x_true = abs(np.random.randn(n,1))
            x_true = x_true / np.linalg.norm(x_true, 1)
            b = A.dot(x_true)
            Q = A.T.dot(A)
            Q_flat = Q.flatten()
            c = -A.T.dot(b)
            G = spdiag([-1.0]*n)
            h = matrix([1.]*n, (n,1))
            U = matrix([1.]*n, (1,n))
            f = matrix(1.0)
            start_time = time.time()
            sol = solvers.qp(matrix(Q), matrix(c), G, h, U, f)
            times_cvxopt.append(time.time() - start_time)
            iters_cvxopt.append(sol['iterations'])
            precision_cvxopt.append(np.linalg.norm(sol['x']-x_true))
            #assert almost_equal(sol['x'], x_true, 1e-6)
            c = c.flatten()

            x0 = np.ones(n) / n

            def proj(x):
                proj_simplex_c(x, 0, n)

            def line_search(x, f, g, x_new, f_new, g_new):
                return line_search_quad_obj(x, f, g, x_new, f_new, g_new, Q_flat, c)

            def obj_c(x, g):
                return quad_obj(x, Q_flat, c, g)

            def obj_np(x, g):
                np.copyto(g, Q.dot(x) + c)
                f = .5 * x.T.dot(g + c)
                return f

            x_true = x_true.flatten()
            start_time = time.time()
            sol = batch.solve(obj_c, proj, line_search, x0)
            times_batch.append(time.time() - start_time)
            precision_batch.append(np.linalg.norm(sol['x']-x_true))
            iters_batch.append(sol['iterations'])
            if n == 300:
                print sol['times']
                print sum(sol['times'])
            #assert almost_equal(sol['x'], x_true, 1e-5)
        print 'CVXOPT'
        print 'times', times_cvxopt
        print 'iterations', iters_cvxopt
        print 'precision', precision_cvxopt
        print 'BATCH'
        print 'times', times_batch
        print 'iterations', iters_batch
        print 'precision', precision_batch


if __name__ == '__main__':
    unittest.main()