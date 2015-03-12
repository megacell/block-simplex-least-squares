import unittest
import time
import numpy as np
from cvxopt import matrix, spdiag, spmatrix, solvers
import sys
sys.path.append('../../../')
from python.c_extensions.c_extensions import (proj_simplex_c,
                                       quad_obj_c, 
                                       line_search_quad_obj_c)
from python.algorithm_utils import (quad_obj_np, 
                                line_search_quad_obj_np, 
                                line_search_exact_quad_obj)
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
        for i,n in enumerate([100, 1000, 10000]): # dimension of features
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
            # assert almost_equal(sol['x'], x_true, 1e-6)
            c = c.flatten()

            x0 = np.ones(n) / n

            def proj(x):
                proj_simplex_c(x, 0, n)
            
            # different types of line search
            def line_search_exact(x, f, g, x_new, f_new, g_new, i):
                return line_search_exact_quad_obj(x, f, g, x_new, f_new, g_new, Q, c) # returns f_new

            def line_search_c(x, f, g, x_new, f_new, g_new, i):
                return line_search_quad_obj_c(x, f, g, x_new, f_new, g_new, Q_flat, c)

            def line_search_np(x, f, g, x_new, f_new, g_new, i):
                return line_search_quad_obj_np(x, f, g, x_new, f_new, g_new, Q, c)
            
            # different types of objective
            def obj_c(x, g):
                return quad_obj_c(x, Q_flat, c, g)

            def obj_np(x, g):
                return quad_obj_np(x, Q, c, g)

            x_true = x_true.flatten()
            start_time = time.time()
            # Batch gradient descent
            #sol = batch.solve(obj_np, proj, line_search_exact, x0)
            sol = batch.solve_BB(obj_np, proj, line_search_exact, x0, prog_tol=1e-12, Q=Q)
            times_batch.append(time.time() - start_time)
            precision_batch.append(np.linalg.norm(sol['x']-x_true))
            iters_batch.append(sol['iterations'])
            print sol['stop']
            if i == 2:
                print 't_proj:', sol['t_proj']
                print 't_obj:', sol['t_obj']
                print 't_line:', sol['t_line']
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
    