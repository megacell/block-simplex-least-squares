
import time
import unittest
import numpy as np
import sys
sys.path.append('../../../')
from python.bsls_utils import (construct_qp_from_least_squares,
                                generate_data)
from python.algorithm_utils import get_solver_parts
import python.BATCH as batch

from openopt import QP

__author__ = 'jeromethai'


class TestSparseGradient(unittest.TestCase):

    def test_sparse_gradient(self):

        times_lbfgs_x_dense = []
        iters_lbfgs_x_dense = []
        error_lbfgs_x_dense = []
        times_cvxopt = []
        iters_cvxopt = []
        error_cvxopt = []

        for i,n in enumerate([100, 1000]):
            m1 = n/2
            A_sparse = 0.9
            data = generate_data(n=n, m1=m1, A_sparse=A_sparse, scale=False)
            A, b, x_true = data['A'], data['b'], data['x_true']
            #print 'norm(Ax_true-b):', np.linalg.norm(A.dot(x_true)-b)
            block_starts = data['block_starts'].astype(int)
            block_sizes = data['blocks'].astype(int)
            #print 'block_sizes:', block_sizes
            #print 'block_starts', block_starts
            G = np.diag([-1.0]*n)
            h = [1.]*n
            U = data['U']
            f = data['f']
            #print 'this is f', f

            Q, c = construct_qp_from_least_squares(A, b)
            min_eig = np.linalg.eig(Q)[0][-1]

            step_size, proj, line_search, obj = get_solver_parts((Q,c), block_starts, min_eig)
            f_min = obj(x_true)
            #print 'check is equal zero:', f_min + 0.5*b.T.dot(b)

            x_init = np.ones(n)
            proj(x_init)
            #print obj(x_init)
            start_time = time.time()
            sol = batch.solve_LBFGS(obj, proj, line_search, x_init)
            print sol['stop']
            times_lbfgs_x_dense.append(time.time() - start_time)
            #print obj(sol['x'])
            #print f_min
            error_lbfgs_x_dense.append(obj(sol['x']) - f_min)
            iters_lbfgs_x_dense.append(sol['iterations'])

            problem = QP(Q, c, A=G, b=h, Aeq=U, beq=f)
            start_time = time.time()
            sol = problem._solve('cvxopt_qp', iprint=0)
            #times_cvxopt.append(sol.elapsed['solver_cputime'])
            times_cvxopt.append(time.time() - start_time)
            iters_cvxopt.append(sol.istop)
            error_cvxopt.append(obj(sol.xf) - f_min)

        print 'times_lbfgs_x_dense', times_lbfgs_x_dense
        print 'error_lbfgs_x_dense', error_lbfgs_x_dense
        print 'iters_lbfgs_x_dense', iters_lbfgs_x_dense
        print 'times_cvxopt', times_cvxopt
        print 'error_cvxopt', error_cvxopt
        print 'iters_cvxopt', iters_cvxopt

if __name__ == '__main__':
    unittest.main()