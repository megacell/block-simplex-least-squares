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
        for i,n in enumerate([100, 500, 1000]): # dimension of features

            # initialize matrices for x variable

            m = 1.5*n # number of measurements
            A = np.random.randn(m, n)
            x_true = abs(np.random.randn(n,1))
            x_true = x_true / np.linalg.norm(x_true, 1)
            b = A.dot(x_true)
            Q = A.T.dot(A)
            c = -A.T.dot(b)
            G = spdiag([-1.0]*n)
            h = matrix([1.]*n, (n,1))
            U = matrix([1.]*n, (1,n))
            f = matrix(1.0)
            x_init = np.ones(n) / n
            
            # initialize for z variable

            x0 

            # define obj, line_search, proj

            def proj(x):
                proj_simplex_c(x, 0, n)
            
            def line_search_exact(x, f, g, x_new, f_new, g_new, i):
                return line_search_exact_quad_obj(x, f, g, x_new, f_new, g_new, Q, c) # returns f_new

            def obj_np(x, g):
                return quad_obj_np(x, Q, c, g)

            # CVXOPT

            start_time = time.time()
            sol = solvers.qp(matrix(Q), matrix(c), G, h, U, f)
            times_cvxopt.append(time.time() - start_time)
            iters_cvxopt.append(sol['iterations'])
            precision_cvxopt.append(np.linalg.norm(sol['x']-x_true))
            c = c.flatten()

            # Bath gradient
            
            x_true = x_true.flatten()
            start_time = time.time()
            #sol = batch.solve(obj_np, proj, line_search_exact, x0)
            sol = batch.solve_BB(obj_np, proj, line_search_exact, x_init, Q=Q)
            times_batch.append(time.time() - start_time)
            precision_batch.append(np.linalg.norm(sol['x']-x_true))
            iters_batch.append(sol['iterations'])
            print sol['stop']
            if i == 2:
                print 't_proj:', sol['t_proj']
                print 't_obj:', sol['t_obj']
                print 't_line:', sol['t_line']

        # display results

        print 'times cvxopt', times_cvxopt
        print 'times batch', times_batch
        print 'iterations cvxopt', iters_cvxopt
        print 'iterations batch', iters_batch
        print 'precision cvxopt', precision_cvxopt
        print 'precision batch', precision_batch


if __name__ == '__main__':
    unittest.main()
    