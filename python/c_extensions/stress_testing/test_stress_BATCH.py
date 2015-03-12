import unittest
import time
import numpy as np
import cvxopt as copt
from cvxopt import matrix, spdiag, spmatrix, solvers
import sys
sys.path.append('../../../')
from python.c_extensions.c_extensions import (proj_simplex_c,
                                       quad_obj_c, 
                                       line_search_quad_obj_c,
                                       isotonic_regression_c)
from python.algorithm_utils import (quad_obj_np, 
                                line_search_quad_obj_np, 
                                line_search_exact_quad_obj)
import python.BATCH as batch
from python.bsls_utils import almost_equal, block_starts_to_N

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
        times_batch2 = []
        iters_cvxopt = []
        iters_batch = []
        iters_batch2 = []
        precision_cvxopt = []
        precision_batch = []
        precision_batch2 = []
        for i,n in enumerate([100, 500, 1000]): # dimension of features

            # initialize matrices for x variable

            m = 1.5*n # number of measurements
            A = np.random.randn(m, n)
            x_true = abs(np.random.randn(n,1))
            x_true = x_true / np.linalg.norm(x_true, 1)
            b = A.dot(x_true)
            Q = A.T.dot(A)
            c = -A.T.dot(b)
            G = copt.spdiag([-1.0]*n)
            h = copt.matrix([1.]*n, (n,1))
            U = copt.matrix([1.]*n, (1,n))
            f = copt.matrix(1.0)
            x_init = np.ones(n) / n
            
            # initialize for z variable

            x0 = np.zeros((n,1))
            x0[-1] = 1.0
            block_starts = np.array([0])
            num_blocks = len(block_starts)
            N = block_starts_to_N(block_starts, n)
            A2 = A.dot(N)
            b2 = b - A.dot(x0)
            Q2 = A2.T.dot(A2)
            c2 = -A2.T.dot(b2)
            z_init = (1. + np.arange(n-num_blocks))/n

            # define obj, line_search, proj
            
            c = c.flatten()
            c2 = c2.flatten()

            def proj(x):
                proj_simplex_c(x, 0, n)

            def proj2(x):
                isotonic_regression_c(x, 0, n-num_blocks)
                np.maximum(0.,x,x)
                np.minimum(1.,x,x)
            
            def line_search_exact(x, f, g, x_new, f_new, g_new, i):
                return line_search_exact_quad_obj(x, f, g, x_new, f_new, g_new, Q, c) # returns f_new
            
            def line_search_exact2(x, f, g, x_new, f_new, g_new, i):
                return line_search_exact_quad_obj(x, f, g, x_new, f_new, g_new, Q2, c2) # returns f_new

            def obj_np(x, g):
                return quad_obj_np(x, Q, c, g) # returns f
    
            def obj_np2(x, g):
                return quad_obj_np(x, Q2, c2, g)

            # CVXOPT

            start_time = time.time()
            sol = copt.solvers.qp(copt.matrix(Q), copt.matrix(c), G, h, U, f)
            times_cvxopt.append(time.time() - start_time)
            iters_cvxopt.append(sol['iterations'])
            precision_cvxopt.append(np.linalg.norm(sol['x']-x_true))
            
            # CPLEX

            # Bath gradient
            
            x_true = x_true.flatten()
            start_time = time.time()
            #sol = batch.solve(obj_np, proj, line_search_exact, x_init)
            sol = batch.solve_BB(obj_np, proj, line_search_exact, x_init, Q=Q)
            times_batch.append(time.time() - start_time)
            precision_batch.append(np.linalg.norm(sol['x']-x_true))
            iters_batch.append(sol['iterations'])
            print sol['stop']
            if i == 2:
                print 't_proj:', sol['t_proj']
                print 't_obj:', sol['t_obj']
                print 't_line:', sol['t_line']


            # # Bath gradient
            
            # start_time = time.time()
            # #sol = batch.solve(obj_np2, proj2, line_search_exact2, x_init)
            # sol = batch.solve_BB(obj_np2, proj2, line_search_exact2, z_init, Q=Q2)
            # times_batch2.append(time.time() - start_time)
            # x_final = N.dot(sol['x']) + x0
            # precision_batch2.append(np.linalg.norm(x_final-x_true))
            # iters_batch2.append(sol['iterations'])
            # print sol['stop']
            # if i == 2:
            #     print 't_proj:', sol['t_proj']
            #     print 't_obj:', sol['t_obj']
            #     print 't_line:', sol['t_line']

        # display results

        print 'times cvxopt', times_cvxopt
        print 'times batch', times_batch
        print 'times batch2', times_batch2
        print 'iterations cvxopt', iters_cvxopt
        print 'iterations batch', iters_batch
        print 'iterations batch2', iters_batch2
        print 'precision cvxopt', precision_cvxopt
        print 'precision batch', precision_batch
        print 'precision batch2', precision_batch2


if __name__ == '__main__':
    unittest.main()
    