import pandas as pd
import unittest
import time
import numpy as np
import cvxopt as copt
import sys
sys.path.append('../../../')
from python.c_extensions.c_extensions import (proj_simplex_c,
                                       quad_obj_c, 
                                       line_search_quad_obj_c,
                                       isotonic_regression_c)
from python.algorithm_utils import (quad_obj_np,
                                    decreasing_step_size,
                                    get_solver_parts)
import python.BATCH as batch
from python.bsls_utils import (almost_equal, 
                                x2z, 
                                qp_to_qp_in_z,
                                random_least_squares,
                                generate_data,
                                construct_qp_from_least_squares)

__author__ = 'jeromethai'

class TestStressBatch(unittest.TestCase):
  

    # def setUp(self):
    #     seed = 237423433
    #     seed = 0
    #     seed = 372983
    #     np.random.seed(seed)
    def test_generate_data(self):
        data = generate_data(m1 = 150)
        #print data['A'][0,:]
        Q, c = construct_qp_from_least_squares(data['A'], data['b'])
        #print Q
        #Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
        print np.linalg.eig(Q)[0][-1]/np.linalg.eig(Q)[0][1]
        block_starts = data['block_starts'].astype(int)
        Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
        print np.linalg.eig(Qz)[0][-1]/np.linalg.eig(Qz)[0][1]


    def save_progress(self, progress, f_min, name, columns):
        iters = len(progress)
        index = pd.MultiIndex.from_tuples(zip([name]*iters, range(iters)))
        for i in range(len(progress)): progress[i][1] -= f_min
        df = pd.DataFrame(progress, index = index, columns = columns)
        return df


    def test_batch(self):
        # test on constrainted least squares problem
        # min ||Ax-b||^2 = x'A'Ax - 2 b'Ax
        # s.t. ||x||_1 = 1, x>=0
        copt.solvers.options['show_progress'] = False
        times_cvxopt = []
        times_batch = []
        times_batch_z = []
        times_bb = []
        times_bb_z = []
        times_lbfgs = []
        times_lbfgs_z = []
        times_md = []

        iters_cvxopt = []
        iters_batch = []
        iters_batch_z = []
        iters_bb = []
        iters_bb_z = []
        iters_lbfgs = []
        iters_lbfgs_z = []
        iters_md = []

        precision_cvxopt = []
        precision_batch = []
        precision_batch_z = []
        precision_bb = []
        precision_bb_z = []
        precision_lbfgs = []
        precision_lbfgs_z = []
        precision_md = []

        dfs = []
        columns = ['time', 'f-f_min']

        for i,n in enumerate([10, 100, 1000]): # dimension of features

            m = 1.5*n # number of measurements
            Q, c, x_true, f_min, min_eig = random_least_squares(m, n, 0.5)
            print 'condition number in x', min_eig / np.linalg.eig(Q)[0][1]
            G = copt.spdiag([-1.0]*n)
            h = copt.matrix([1.]*n, (n,1))
            U = copt.matrix([1.]*n, (1,n))
            f = copt.matrix(1.0)
            block_starts = np.array([0])
            num_blocks = len(block_starts)
            step_size, proj, line_search, obj = get_solver_parts((Q, c), min_eig)

            # converts into z-variable
            Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
            f_min_z = f_min - f0
            min_eig_z = np.linalg.eig(Qz)[0][-1]
            print 'condition number in z', min_eig_z / np.linalg.eig(Qz)[0][1]
            step_size_z, proj_z, line_search_z, obj_z = get_solver_parts((Qz, cz), min_eig_z, True)


            # CVXOPT
            start_time = time.time()
            sol = copt.solvers.qp(copt.matrix(Q), copt.matrix(c), G, h, U, f)
            times_cvxopt.append(time.time() - start_time)
            iters_cvxopt.append(sol['iterations'])
            #precision_cvxopt.append(np.linalg.norm(sol['x']-x_true))
            precision_cvxopt.append(obj(np.array(sol['x']).flatten()) - f_min)

            # CPLEX

            # Batch gradient descent in x
            
            x_init = np.ones(n) / n
            start_time = time.time()
            #sol = batch.solve(obj_np, proj, line_search, x_init)
            sol = batch.solve(obj, proj, step_size, x_init, line_search)
            times_batch.append(time.time() - start_time)
            #precision_batch.append(np.linalg.norm(sol['x']-x_true))
            precision_batch.append(obj(sol['x']) - f_min)
            iters_batch.append(sol['iterations'])
            dfs.append(self.save_progress(sol['progress'], f_min, 'batch_x_'+str(i), columns))

            # Batch gradient descent in z

            x_init = np.ones(n) / n
            z_init = x2z(x_init, block_starts=block_starts)
            start_time = time.time()
            sol = batch.solve(obj_z, proj_z, step_size_z, z_init, line_search_z)
            times_batch_z.append(time.time() - start_time)
            #x_final = N.dot(sol['x']) + x0
            #precision_batch_z.append(np.linalg.norm(x_final-x_true))
            precision_batch_z.append(obj_z(sol['x']) - f_min_z)
            iters_batch_z.append(sol['iterations'])
            dfs.append(self.save_progress(sol['progress'], f_min_z, 'batch_z_'+str(i), columns))

            # gradient descent with BB step in x

            x_init = np.ones(n) / n
            start_time = time.time()
            #sol = batch.solve(obj_np, proj, line_search, x_init)
            sol = batch.solve_BB(obj, proj, line_search, x_init)
            times_bb.append(time.time() - start_time)
            #precision_batch.append(np.linalg.norm(sol['x']-x_true))
            precision_bb.append(obj(sol['x']) - f_min)
            iters_bb.append(sol['iterations'])
            dfs.append(self.save_progress(sol['progress'], f_min, 'bb_x_'+str(i), columns))

            # gradient descent with BB step in z

            x_init = np.ones(n) / n
            z_init = x2z(x_init, block_starts=block_starts)
            start_time = time.time()
            sol = batch.solve_BB(obj_z, proj_z, line_search_z, z_init)
            times_bb_z.append(time.time() - start_time)
            #x_final = N.dot(sol['x']) + x0
            #precision_batch_z.append(np.linalg.norm(x_final-x_true))
            precision_bb_z.append(obj_z(sol['x']) - f_min_z)
            iters_bb_z.append(sol['iterations'])
            dfs.append(self.save_progress(sol['progress'], f_min_z, 'bb_z_'+str(i), columns))

            # l-BFGS in x
            
            x_init = np.ones(n) / n
            start_time = time.time()
            sol = batch.solve_LBFGS(obj, proj, line_search, x_init)
            times_lbfgs.append(time.time() - start_time)
            #precision_lbfgs.append(np.linalg.norm(sol['x']-x_true))
            precision_lbfgs.append(obj(sol['x']) - f_min)
            iters_lbfgs.append(sol['iterations'])
            dfs.append(self.save_progress(sol['progress'], f_min, 'lbfgs_x_'+str(i), columns))

            # l-BFGS in z
            
            x_init = np.ones(n) / n
            z_init = x2z(x_init, block_starts=block_starts)
            start_time = time.time()
            #sol = batch.solve(obj_npz, projz, line_searchz, z_init)
            sol = batch.solve_LBFGS(obj_z, proj_z, line_search_z, z_init)
            times_lbfgs_z.append(time.time() - start_time)
            #x_final = N.dot(sol['x']) + x0
            #precision_lbfgs_z.append(np.linalg.norm(x_final-x_true))
            precision_lbfgs_z.append(obj_z(sol['x']) - f_min_z)
            iters_lbfgs_z.append(sol['iterations'])
            dfs.append(self.save_progress(sol['progress'], f_min_z, 'lbfgs_z_'+str(i), columns))

            # mirror descent

            x_init = np.ones(n) / n
            start_time = time.time()
            def step_size(i):
                return decreasing_step_size(i, 1.0, min_eig/16.0)
            sol = batch.solve_MD(obj,block_starts, step_size, x_init)
            times_md.append(time.time() - start_time)
            #precision_batch.append(np.linalg.norm(sol['x']-x_true))
            precision_md.append(obj(sol['x']) - f_min)
            iters_md.append(sol['iterations'])
            dfs.append(self.save_progress(sol['progress'], f_min, 'md_x_'+str(i), columns))

        progress = pd.concat(dfs)
        progress.save('progress.pkl')

        # display results

        print 'times cvxopt', times_cvxopt
        print 'times batch', times_batch
        print 'times batch_z', times_batch_z
        print 'times bb', times_bb
        print 'times bb_z', times_bb_z
        print 'times lbfgs', times_lbfgs
        print 'times lbfgs_z', times_lbfgs_z
        print 'times md', times_md

        print 'iterations cvxopt', iters_cvxopt
        print 'iterations batch', iters_batch
        print 'iterations batch_z', iters_batch_z
        print 'iterations bb', iters_bb
        print 'iterations bb_z', iters_bb_z
        print 'iterations lbfgs', iters_lbfgs
        print 'iterations lbfgs_z', iters_lbfgs_z
        print 'iterations md', iters_md

        print 'precision cvxopt', precision_cvxopt
        print 'precision batch', precision_batch
        print 'precision batch_z', precision_batch_z
        print 'precision bb', precision_bb
        print 'precision bb_z', precision_bb_z
        print 'precision lbfgs', precision_lbfgs
        print 'precision lbfgs_z', precision_lbfgs_z
        print 'precision md', precision_md


if __name__ == '__main__':
    unittest.main()
    