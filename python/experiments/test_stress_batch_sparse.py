import pdb
import pandas as pd
import time
import unittest
import numpy as np
import sys
sys.path.append('../../')
import scipy
from python.bsls_utils import (construct_qp_from_least_squares,
                                generate_data,
                                ls_to_ls_in_z,
                                qp_to_qp_in_z,
                                x2z)
from python.algorithm_utils import get_solver_parts, save_progress
import python.BATCH as batch

from openopt import QP

__author__ = 'jeromethai'


class TestSparseGradient(unittest.TestCase):

    def test_sparse_gradient(self):

        times_lbfgs_x_dense = []
        iters_lbfgs_x_dense = []
        error_lbfgs_x_dense = []

        times_lbfgs_x_sparse = []
        iters_lbfgs_x_sparse = []
        error_lbfgs_x_sparse = []

        times_lbfgs_z_dense = []
        iters_lbfgs_z_dense = []
        error_lbfgs_z_dense = []

        times_lbfgs_z_sparse = []
        iters_lbfgs_z_sparse = []
        error_lbfgs_z_sparse = []

        times_cvxopt = []
        iters_cvxopt = []
        error_cvxopt = []

        # initialize database
        dfs = []

        # generate a least squares well-conditioned in z
        in_z = False

        for i,n in enumerate([100, 1000, 2000]):
            m1 = n/10
            m2 = n/10
            A_sparse = 0.9
            data = generate_data(n=n, m1=m1, A_sparse=A_sparse, scale=False, m2=m2, in_z=in_z)
            #data = scipy.io.loadmat('test_mat.mat')
            A = data['A']
            b = np.squeeze(data['b'])
            x_true = np.squeeze(data['x_true'])

            #print 'norm(Ax_true-b):', np.linalg.norm(A.dot(x_true)-b)
            block_starts = np.squeeze(data['block_starts']).astype(int)
            block_sizes = np.squeeze(data['blocks']).astype(int)

            n = x_true.shape[0]
            m1 = b.shape[0]           # number of measurements
            m2 = block_sizes.shape[0] # number of blocks

            Az, bz, N, x0 = ls_to_ls_in_z(A, b, block_starts)
            #print 'block_sizes:', block_sizes
            #print 'block_starts', block_starts
            G = np.diag([-1.0]*n)
            h = [1.]*n
            U = data['U']
            f = np.squeeze(data['f'])
            #print 'this is f', f

            Q, c = construct_qp_from_least_squares(A, b)
            Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
            w = np.linalg.eig(Q)[0]
            min_eig = w[-1]
            print 'min_eig:', min_eig
            wz = np.linalg.eig(Qz)[0]
            min_eig_z = wz[-1]
            print 'min_eig_z', min_eig_z

            step_size, proj, line_search, obj = get_solver_parts((Q,c), block_starts, min_eig)
            _, _, line_search_sparse, obj_sparse = get_solver_parts((A,b), block_starts, min_eig, is_sparse=True)
            step_size_z, proj_z, line_search_z, obj_z = get_solver_parts((Qz, cz), block_starts, min_eig_z, True)
            _, _, line_search_sparse_z, obj_sparse_z = get_solver_parts((Az,bz), block_starts, min_eig_z, is_sparse=True)


            f_min = obj(x_true)
            f_min_z = f_min - f0
            #print 'check is equal zero:', f_min + 0.5*b.T.dot(b)

            # lbfgs in x dense

            x_init = np.ones(n)
            proj(x_init)
            #print obj(x_init)
            start_time = time.time()
            sol = batch.solve_LBFGS(obj, proj, line_search, x_init)
            times_lbfgs_x_dense.append(time.time() - start_time)
            error_lbfgs_x_dense.append(obj(sol['x']) - f_min)
            iters_lbfgs_x_dense.append(sol['iterations'])
            dfs.append(save_progress(sol['progress'], f_min, 'lbfgs_x_dense_'+str(i)))

            # lbfgs in x sparse

            x_init = np.ones(n)
            proj(x_init)
            start_time = time.time()
            sol = batch.solve_LBFGS(obj_sparse, proj, line_search_sparse, x_init)
            times_lbfgs_x_sparse.append(time.time() - start_time)
            error_lbfgs_x_sparse.append(obj(sol['x']) - f_min)
            iters_lbfgs_x_sparse.append(sol['iterations'])
            dfs.append(save_progress(sol['progress'], 0.0, 'lbfgs_x_sparse_'+str(i)))


            # lbfgs in z dense

            x_init = np.ones(n)
            proj(x_init)
            z_init = x2z(x_init, block_starts=block_starts)
            start_time = time.time()
            sol = batch.solve_LBFGS(obj_z, proj_z, line_search_z, z_init)
            times_lbfgs_z_dense.append(time.time() - start_time)
            error_lbfgs_z_dense.append(obj_z(sol['x']) - f_min_z)
            iters_lbfgs_z_dense.append(sol['iterations'])
            dfs.append(save_progress(sol['progress'], f_min_z, 'lbfgs_z_dense_'+str(i)))


            # lbfgs in z sparse

            x_init = np.ones(n)
            proj(x_init)
            z_init = x2z(x_init, block_starts=block_starts)
            start_time = time.time()
            sol = batch.solve_LBFGS(obj_sparse_z, proj_z, line_search_sparse_z, z_init)
            times_lbfgs_z_sparse.append(time.time() - start_time)
            error_lbfgs_z_sparse.append(obj_z(sol['x']) - f_min_z)
            iters_lbfgs_z_sparse.append(sol['iterations'])
            dfs.append(save_progress(sol['progress'], 0.0, 'lbfgs_z_sparse_'+str(i)))


            # cvxopt

            # problem = QP(Q, c, A=G, b=h, Aeq=U, beq=f)
            # start_time = time.time()
            # sol = problem._solve('cvxopt_qp', iprint=0)
            # #times_cvxopt.append(sol.elapsed['solver_cputime'])
            # times_cvxopt.append(time.time() - start_time)
            # iters_cvxopt.append(sol.istop)
            # error_cvxopt.append(obj(sol.xf) - f_min)

        progress = pd.concat(dfs)
        progress.save('results/progress_sparse.pkl')

        print 'times_lbfgs_x_dense', times_lbfgs_x_dense
        print 'error_lbfgs_x_dense', error_lbfgs_x_dense
        print 'iters_lbfgs_x_dense', iters_lbfgs_x_dense

        print 'times_lbfgs_x_sparse', times_lbfgs_x_sparse
        print 'error_lbfgs_x_sparse', error_lbfgs_x_sparse
        print 'iters_lbfgs_x_sparse', iters_lbfgs_x_sparse

        print 'times_lbfgs_z_dense', times_lbfgs_z_dense
        print 'error_lbfgs_z_dense', error_lbfgs_z_dense
        print 'iters_lbfgs_z_dense', iters_lbfgs_z_dense

        print 'times_lbfgs_z_sparse', times_lbfgs_z_sparse
        print 'error_lbfgs_z_sparse', error_lbfgs_z_sparse
        print 'iters_lbfgs_z_sparse', iters_lbfgs_z_sparse

        print 'times_cvxopt', times_cvxopt
        print 'error_cvxopt', error_cvxopt
        print 'iters_cvxopt', iters_cvxopt

if __name__ == '__main__':
    unittest.main()
