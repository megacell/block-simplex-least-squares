import unittest
import time
import numpy as np
import sys

from openopt import QP
sys.path.append('../../../')
from python.c_extensions.c_extensions import (proj_simplex_c,
                                       quad_obj_c,
                                       line_search_quad_obj_c,
                                       isotonic_regression_c)
from python.algorithm_utils import (quad_obj_np,
                                    decreasing_step_size,
                                    line_search_quad_obj_np)
import python.BATCH as batch
from python.bsls_utils import almost_equal, x2z, qp_to_qp_in_z

__author__ = 'jeromethai'

class TestStressBatch(unittest.TestCase):


    def setUp(self):
        seed = 237423433
        #seed = 0
        np.random.seed(seed)


    def test_batch(self):
        # test on constrainted least squares problem
        # min ||Ax-b||^2 = x'A'Ax - 2 b'Ax
        # s.t. ||x||_1 = 1, x>=0
        #copt.solvers.options['show_progress'] = False
        times_cvxopt = []
        times_cplex = []
        times_batch = []
        times_batchz = []
        times_lbfgs = []
        times_lbfgsz = []
        iters_cvxopt = []
        iters_cplex = []
        iters_batch = []
        iters_batchz = []
        iters_lbfgs = []
        iters_lbfgsz = []
        precision_cvxopt = []
        precision_cplex = []
        precision_batch = []
        precision_batchz = []
        precision_lbfgs = []
        precision_lbfgsz = []

        for i,n in enumerate([100, 1000, 2000]): # dimension of features

            # initialize matrices for x variable

            m = 1.5*n # number of measurements
            A = np.random.randn(m, n)
            x_true = abs(np.random.randn(n,1))
            x_true = x_true / np.linalg.norm(x_true, 1)
            b = A.dot(x_true)
            Q = A.T.dot(A)
            c = (-A.T.dot(b)).flatten()
            G = np.diag([-1.0]*n)
            h = [1.]*n
            U = [1.]*n
            f = np.matrix(1.0)
            block_starts = np.array([0])
            num_blocks = len(block_starts)

            # converts into z-variable
            Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)

            # define obj, line_search, proj
            def proj(x):
                proj_simplex_c(x, 0, n)

            def projz(x):
                isotonic_regression_c(x, 0, n-num_blocks)
                np.maximum(0.,x,x)
                np.minimum(1.,x,x)

            def step_size(i):
                t0 = 1.0
                alpha = 8.0
                return decreasing_step_size(i, t0, alpha)

            def step_size_z(i):
                t0 = 1.0
                alpha = 8.0
                return decreasing_step_size(i, t0, alpha)

            def line_search(x, f, g, x_new, f_new, g_new, i):
                return line_search_quad_obj_np(x, f, g, x_new, f_new, g_new, Q, c) # returns f_new

            def line_searchz(x, f, g, x_new, f_new, g_new, i):
                return line_search_quad_obj_np(x, f, g, x_new, f_new, g_new, Qz, cz) # returns f_new

            def obj_np(x, g):
                return quad_obj_np(x, Q, c, g) # returns f

            def obj_npz(x, g):
                return quad_obj_np(x, Qz, cz, g)



            # CVXOPT
            #sol = copt.solvers.qp(copt.matrix(Q), copt.matrix(c), G, h, U, f)
            problem = QP(Q, c, A=G, b=h, Aeq=U, beq=f)
            sol = problem._solve('cvxopt_qp', iprint=0)
            times_cvxopt.append(sol.elapsed['solver_cputime'])
            iters_cvxopt.append(sol.istop)
            precision_cvxopt.append(np.linalg.norm(sol.xf - x_true))

            # CPLEX
            problem = QP(Q, c, A=G, b=h, Aeq=U, beq=f)
            sol = problem._solve('cplex', iprint=0)
            times_cplex.append(sol.elapsed['solver_cputime'])
            iters_cplex.append(sol.istop)
            precision_cplex.append(np.linalg.norm(sol.xf - x_true))

            # Batch gradient in x

            x_init = np.ones(n) / n
            x_true = x_true.flatten()
            start_time = time.time()
            #sol = batch.solve(obj_np, proj, line_search, x_init)
            sol = batch.solve(obj_np, proj, step_size, x_init, max_iter=1000)
            times_batch.append(time.time() - start_time)
            precision_batch.append(np.linalg.norm(sol['x']-x_true))
            iters_batch.append(sol['iterations'])
            print sol['stop']
            if i == 2:
                print 't_proj:', sol['t_proj']
                print 't_obj:', sol['t_obj']
                print 't_line:', sol['t_line']

            # Batch gradient in z

            x_init = np.ones(n) / n
            z_init = x2z(x_init, block_starts=block_starts)
            start_time = time.time()
            sol = batch.solve(obj_npz, projz, step_size_z, z_init, max_iter=1000)
            times_batchz.append(time.time() - start_time)
            x_final = N.dot(sol['x']) + x0
            precision_batchz.append(np.linalg.norm(x_final-x_true))
            iters_batchz.append(sol['iterations'])
            if i == 2:
                print 't_proj:', sol['t_proj']
                print 't_obj:', sol['t_obj']
                print 't_line:', sol['t_line']

            # l-BFGS in x

            x_init = np.ones(n) / n
            start_time = time.time()
            #sol = batch.solve(obj_npz, projz, line_searchz, z_init)
            sol = batch.solve_LBFGS(obj_np, proj, line_search, x_init)
            times_lbfgs.append(time.time() - start_time)
            precision_lbfgs.append(np.linalg.norm(sol['x']-x_true))
            iters_lbfgs.append(sol['iterations'])
            #print sol['stop']
            # if i == 2:
            #     print 't_proj:', sol['t_proj']
            #     print 't_obj:', sol['t_obj']
            #     print 't_line:', sol['t_line']

            # l-BFGS in z

            x_init = np.ones(n) / n
            z_init = x2z(x_init, block_starts=block_starts)
            start_time = time.time()
            #sol = batch.solve(obj_npz, projz, line_searchz, z_init)
            sol = batch.solve_LBFGS(obj_npz, projz, line_searchz, z_init)
            print sol['stop']
            times_lbfgsz.append(time.time() - start_time)
            x_final = N.dot(sol['x']) + x0
            precision_lbfgsz.append(np.linalg.norm(x_final-x_true))
            iters_lbfgsz.append(sol['iterations'])
            #print sol['stop']
            # if i == 2:
            #     print 't_proj:', sol['t_proj']
            #     print 't_obj:', sol['t_obj']
            #     print 't_line:', sol['t_line']

        # display results

        print 'times cvxopt', times_cvxopt
        print 'times cplex', times_cplex
        print 'times batch', times_batch
        print 'times batchz', times_batchz
        print 'times lbfgs', times_lbfgs
        print 'times lbfgsz', times_lbfgsz
        print 'iterations cvxopt', iters_cvxopt
        print 'iterations cplex', iters_cplex
        print 'iterations batch', iters_batch
        print 'iterations batchz', iters_batchz
        print 'iterations lbfgs', iters_lbfgs
        print 'iterations lbfgsz', iters_lbfgsz
        print 'precision cvxopt', precision_cvxopt
        print 'precision cplex', precision_cplex
        print 'precision batch', precision_batch
        print 'precision batchz', precision_batchz
        print 'precision lbfgs', precision_lbfgs
        print 'precision lbfgsz', precision_lbfgsz


if __name__ == '__main__':
    unittest.main()
