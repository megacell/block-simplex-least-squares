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
                                x2z,
                                coherence)
from python.algorithm_utils import get_solver_parts, save_progress
from python.data_utils import (load_and_process,
                               process_data,
                               remove_measurement,
                               aggregate,
                               clean_progress)
import python.BATCH as batch

from openopt import QP
import cvxopt as copt

__author__ = 'jeromethai'


class TestSparseGradient(unittest.TestCase):

    def test_sparse_gradient(self):

        times_bb_x_dense = []
        iters_bb_x_dense = []
        error_bb_x_dense = []

        times_bb_x_sparse = []
        iters_bb_x_sparse = []
        error_bb_x_sparse = []

        times_bb_z_dense = []
        iters_bb_z_dense = []
        error_bb_z_dense = []

        times_bb_z_sparse = []
        iters_bb_z_sparse = []
        error_bb_z_sparse = []

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

        # choose the experiment type
        experiment = 3 # 1, 2 or 3

        for i,n in enumerate([1000]):

            print 'experiment', i
            m1 = 10 # number of measurements
            m2 = 50 # number of blocks

            A_sparse = 0.7
            #distribution = 'affine'
            distribution = 'uniform'
            #distribution = 'aggregated'

            if experiment == 1:
                data = generate_data(n=n, m1=m1, A_sparse=A_sparse, m2=m2, distribution=distribution)

            if experiment == 2:
                data = scipy.io.loadmat('data/test_mat.mat')
                A, b, U, f, x_true = data['A'], data['b'], data['U'], data['f'], data['x_true']
                data = process_data(A, b, U, f, x_true)

            if experiment == 3:
                data = load_and_process('data/small_network_data.pkl')
                f = data['f']

            A = data['A']
            b = np.squeeze(data['b'])
            x_true = np.squeeze(data['x_true'])
            U = data['U']
            f = np.squeeze(data['f'])
            block_starts = np.squeeze(data['block_starts']).astype(int)
            block_sizes = np.squeeze(data['block_sizes']).astype(int)

            aggregate(A, x_true, block_starts)
            #assert np.linalg.norm(U.dot(x_true)-f) < 1e-5, "Ux!=f after permuting"
            #assert np.linalg.norm(A.dot(x_true)-b) < 1e-5, "Ax!=b after permuting"

            n = x_true.shape[0]
            m1 = b.shape[0]           # number of measurements
            m2 = block_sizes.shape[0] # number of blocks

            G = np.diag([-1.0]*n)
            h = [1.]*n
            #h = np.zeros(n)

            #print 'norm(Ax_true-b):', np.linalg.norm(A.dot(x_true)-b)

            Az, bz, N, x0 = ls_to_ls_in_z(A, b, block_starts, f=f)
            #print 'block_sizes:', block_sizes
            #print 'block_starts', block_starts

            #print 'this is f', f

            Q, c = construct_qp_from_least_squares(A, b)
            Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts, f=f)
            w = np.linalg.eig(Q)[0]
            min_eig = w[-1]
            print 'min_eig:', min_eig
            print 'max_eig:', w[0]
            wz = np.linalg.eig(Qz)[0]
            min_eig_z = wz[-1]
            print 'min_eig_z', min_eig_z
            print 'max_eig_z:', wz[0]

            #print 'sum rows of A'
            #m, n = A.shape
            #print [np.sum(A[i,:]) for i in range(m)]
            print 'coherence of A:', coherence(A)
            print 'sparsity of A:', np.sum(abs(A)), A.shape[0]*A.shape[1]

            #print 'sum rows of Az'
            #m, n = Az.shape
            #print [np.sum(Az[i,:]) for i in range(m)]
            print 'coherence of Az:', coherence(Az)
            print 'sparsity of Az:', np.sum(Az!=0.0), Az.shape[0]*Az.shape[1]

            #pdb.set_trace()

            step_size, proj, line_search, obj = get_solver_parts((Q,c), block_starts, 1e-4, f=f)
            _, _, line_search_sparse, obj_sparse = get_solver_parts((A,b), block_starts, 100, is_sparse=True, f=f)
            step_size_z, proj_z, line_search_z, obj_z = get_solver_parts((Qz, cz), block_starts, 1e-8, True, f=f)
            _, _, line_search_sparse_z, obj_sparse_z = get_solver_parts((Az,bz), block_starts, 100, is_sparse=True, f=f)


            f_min = obj(x_true)
            f_min_z = f_min - f0
            #print 'check is equal zero:', f_min + 0.5*b.T.dot(b)

            x_init = np.random.rand(n)
            proj(x_init)
            z_init = x2z(x_init, block_starts=block_starts)

            # cvxopt
            # from cvxopt import sparse, matrix, solvers
            # Q2 = sparse(matrix(Q))
            # c2 = sparse(matrix(c))
            # #problem = QP(Q2, c2, A=G, b=h, Aeq=U, beq=f)
            # start_time = time.time()
            # sol=solvers.qp(Q2, c2, A=G, b=h, A=U, b=f)
            # #times_cvxopt.append(sol.elapsed['solver_cputime'])
            # times_cvxopt.append(time.time() - start_time)
            # iters_cvxopt.append(sol.istop)
            # error_cvxopt.append(obj(sol.xf) - f_min)

            alphas = []

            # batch in x sparse
            start_time = time.time()
            sol = batch.solve(obj_sparse, proj, step_size, x_init, line_search_sparse)
            dfs.append(save_progress(sol['progress'], 0.0, 'batch_x_sparse_'+str(i)))
            x, y = zip(*sol['progress'])
            x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
            alphas.append(alpha)

            # batch in z sparse            
            start_time = time.time()
            sol = batch.solve(obj_sparse_z, proj_z, step_size_z, z_init, line_search_sparse_z)
            dfs.append(save_progress(sol['progress'], 0.0, 'batch_z_sparse_'+str(i)))
            x, y = zip(*sol['progress'])
            x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
            alphas.append(alpha)

            # batch in x dense
            start_time = time.time()
            sol = batch.solve(obj, proj, step_size, x_init, line_search)
            dfs.append(save_progress(sol['progress'], f_min, 'batch_x_dense_'+str(i)))
            x, y = zip(*sol['progress'])
            x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
            alphas.append(alpha)

            # batch in z dense
            start_time = time.time()
            sol = batch.solve(obj_z, proj_z, step_size_z, z_init, line_search_z)
            dfs.append(save_progress(sol['progress'], f_min_z, 'batch_z_dense_'+str(i)))
            x, y = zip(*sol['progress'])
            x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
            alphas.append(alpha)

            # bb in x dense
            start_time = time.time()
            sol = batch.solve_BB(obj, proj, line_search, x_init)
            times_bb_x_dense.append(time.time() - start_time)
            error_bb_x_dense.append(obj(sol['x']) - f_min)
            iters_bb_x_dense.append(sol['iterations'])
            dfs.append(save_progress(sol['progress'], f_min, 'bb_x_dense_'+str(i)))
            x, y = zip(*sol['progress'])
            x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
            alphas.append(alpha)

            # bb in x sparse
            start_time = time.time()
            sol = batch.solve_BB(obj_sparse, proj, line_search_sparse, x_init)
            times_bb_x_sparse.append(time.time() - start_time)
            error_bb_x_sparse.append(obj(sol['x']) - f_min)
            iters_bb_x_sparse.append(sol['iterations'])
            dfs.append(save_progress(sol['progress'], 0.0, 'bb_x_sparse_'+str(i)))
            x, y = zip(*sol['progress'])
            x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
            alphas.append(alpha)

            # bb in z dense
            start_time = time.time()
            sol = batch.solve_BB(obj_z, proj_z, line_search_z, z_init)
            times_bb_z_dense.append(time.time() - start_time)
            error_bb_z_dense.append(obj_z(sol['x']) - f_min_z)
            iters_bb_z_dense.append(sol['iterations'])
            dfs.append(save_progress(sol['progress'], f_min_z, 'bb_z_dense_'+str(i)))
            x, y = zip(*sol['progress'])
            x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
            alphas.append(alpha)

            # bb in z sparse
            start_time = time.time()
            sol = batch.solve_BB(obj_sparse_z, proj_z, line_search_sparse_z, z_init)
            times_bb_z_sparse.append(time.time() - start_time)
            error_bb_z_sparse.append(obj_z(sol['x']) - f_min_z)
            iters_bb_z_sparse.append(sol['iterations'])
            dfs.append(save_progress(sol['progress'], 0.0, 'bb_z_sparse_'+str(i)))
            x, y = zip(*sol['progress'])
            x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
            alphas.append(alpha)

            # lbfgs in x dense
            start_time = time.time()
            sol = batch.solve_LBFGS(obj, proj, line_search, x_init)
            times_lbfgs_x_dense.append(time.time() - start_time)
            error_lbfgs_x_dense.append(obj(sol['x']) - f_min)
            iters_lbfgs_x_dense.append(sol['iterations'])
            dfs.append(save_progress(sol['progress'], f_min, 'lbfgs_x_dense_'+str(i)))
            x, y = zip(*sol['progress'])
            x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
            alphas.append(alpha)

            # lbfgs in x sparse
            start_time = time.time()
            sol = batch.solve_LBFGS(obj_sparse, proj, line_search_sparse, x_init)
            times_lbfgs_x_sparse.append(time.time() - start_time)
            error_lbfgs_x_sparse.append(obj(sol['x']) - f_min)
            iters_lbfgs_x_sparse.append(sol['iterations'])
            dfs.append(save_progress(sol['progress'], 0.0, 'lbfgs_x_sparse_'+str(i)))
            x, y = zip(*sol['progress'])
            x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
            alphas.append(alpha)

            # lbfgs in z dense
            start_time = time.time()
            sol = batch.solve_LBFGS(obj_z, proj_z, line_search_z, z_init)
            times_lbfgs_z_dense.append(time.time() - start_time)
            error_lbfgs_z_dense.append(obj_z(sol['x']) - f_min_z)
            iters_lbfgs_z_dense.append(sol['iterations'])
            dfs.append(save_progress(sol['progress'], f_min_z, 'lbfgs_z_dense_'+str(i)))
            x, y = zip(*sol['progress'])
            x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
            alphas.append(alpha)

            # lbfgs in z sparse
            start_time = time.time()
            sol = batch.solve_LBFGS(obj_sparse_z, proj_z, line_search_sparse_z, z_init)
            times_lbfgs_z_sparse.append(time.time() - start_time)
            error_lbfgs_z_sparse.append(obj_z(sol['x']) - f_min_z)
            iters_lbfgs_z_sparse.append(sol['iterations'])
            dfs.append(save_progress(sol['progress'], 0.0, 'lbfgs_z_sparse_'+str(i)))
            x, y = zip(*sol['progress'])
            x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
            alphas.append(alpha)

        progress = pd.concat(dfs)
        progress.save('results/progress_sparse.pkl')
        # print alphas
        
        # print 'times_lbfgs_x_dense', times_lbfgs_x_dense
        # print 'error_lbfgs_x_dense', error_lbfgs_x_dense
        # print 'iters_lbfgs_x_dense', iters_lbfgs_x_dense

        # print 'times_lbfgs_x_sparse', times_lbfgs_x_sparse
        # print 'error_lbfgs_x_sparse', error_lbfgs_x_sparse
        # print 'iters_lbfgs_x_sparse', iters_lbfgs_x_sparse

        # print 'times_lbfgs_z_dense', times_lbfgs_z_dense
        # print 'error_lbfgs_z_dense', error_lbfgs_z_dense
        # print 'iters_lbfgs_z_dense', iters_lbfgs_z_dense

        # print 'times_lbfgs_z_sparse', times_lbfgs_z_sparse
        # print 'error_lbfgs_z_sparse', error_lbfgs_z_sparse
        # print 'iters_lbfgs_z_sparse', iters_lbfgs_z_sparse

        # print 'times_cvxopt', times_cvxopt
        # print 'error_cvxopt', error_cvxopt
        # print 'iters_cvxopt', iters_cvxopt


if __name__ == '__main__':
    unittest.main()
