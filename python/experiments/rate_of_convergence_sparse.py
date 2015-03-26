"""
In this experiment, we aim at comparing the rate of convergence
between different gradient, bb, and lbfgs as the nember of measurements
decreases 
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from algorithm_utils import get_solver_parts, save_progress
import BATCH as batch
from bsls_utils import (x2z, 
                        qp_to_qp_in_z, 
                        generate_data, 
                        coherence,
                        ls_to_ls_in_z,
                        construct_qp_from_least_squares)
from data_utils import clean_progress, aggregate

__author__ = 'jeromethai'

algorithms = ['raw_x', 'raw_z', 'raw_x_sparse', 'raw_z_sparse', 
                'bb_x', 'bb_z', 'bb_x_sparse', 'bb_z_sparse',
                'lbfgs_x', 'lbfgs_z', 'lbfgs_x_sparse', 'lbfgs_z_sparse']

n = 1000
num_blocks = 50 # number of blocks
trials = 1
lasso = False

measurement_ratios = [0.01, 0.03, 0.1, 0.3, 1.0]
#measurement_ratios = [0.3, 1.0]

distributions = ['aggregated', 'uniform']
distributions = ['uniform']


def get_rate_of_convergence():

    results = []
    coherences = []

    for distribution in distributions:
        print 'distribution', distribution

        dfs2 = []
        dfs3 = []

        for ratio in measurement_ratios:
            m = int(ratio*n) # number of measurements
            print 'number of measurements', m

            # generate random and process data
            data = generate_data(n=n, m1=m, A_sparse=0.7, m2=num_blocks, 
                                    distribution=distribution)
            A, b, x_true = data['A'], np.squeeze(data['b']), np.squeeze(data['x_true'])
            U, f = data['U'], np.squeeze(data['f'])
            block_starts = np.squeeze(data['block_starts']).astype(int)
            block_sizes = np.squeeze(data['block_sizes']).astype(int)
            aggregate(A, x_true, block_starts)
            #import pdb; pdb.set_trace()
            # contruct associated QP, least-squares in z, QP in z
            Q, c = construct_qp_from_least_squares(A, b)
            Az, bz, N, x0 = ls_to_ls_in_z(A, b, block_starts, f=f)
            Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts, f=f)

            # get condition number and coherence
            w = np.linalg.eig(Q)[0]
            wz = np.linalg.eig(Qz)[0]
            print 'condition number in x', w[0]/w[-1]
            print 'condition number in z', wz[0]/wz[-1]
            print 'max eigen value in x', w[0]
            print 'max eigen value in z', wz[0]            
            c_x = coherence(A)
            c_z = coherence(Az)
            dfs3.append([c_x[0], c_x[1], c_z[0], c_z[1]])
            print 'coherence in x', c_x[0], c_x[1]
            print 'coherence in z', c_z[0], c_z[1]

            # get solver parts for which the rate where the rate alpha
            # for the step size has been optimized
            step_size, proj, line_search, obj = get_solver_parts((Q, c), 
                    block_starts, 10., f=f)
            step_size_z, proj_z, line_search_z, obj_z = get_solver_parts((Qz, cz), 
                    block_starts, 10., True, f=f)
            _, _, line_search_sparse, obj_sparse = get_solver_parts((A,b), block_starts, 10., is_sparse=True, f=f)
            _, _, line_search_sparse_z, obj_sparse_z = get_solver_parts((Az,bz), block_starts, 10., is_sparse=True, f=f)
            f_min = obj(x_true)
            f_min_z = f_min - f0

            alphas = [[] for i in range(12)]

            print 'trials'

            for i in range(trials):
                print i+1

                # initialize
                x_init = np.random.rand(n)
                proj(x_init)
                z_init = x2z(x_init, block_starts=block_starts)

                #raw gradient in x
                sol = batch.solve(obj, proj, step_size, x_init, line_search)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
                alphas[0].append(alpha)
                # plt.plot(x, log_y)

                # raw gradient in z
                sol = batch.solve(obj_z, proj_z, step_size_z, z_init, line_search_z)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min_z)
                alphas[1].append(alpha)
                # plt.plot(x, log_y)
                # plt.show()

                # raw sparse gradient in x
                sol = batch.solve(obj_sparse, proj, step_size, x_init, line_search_sparse)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y))
                alphas[2].append(alpha)
                # plt.plot(x, log_y)

                # raw sparse gradient in z
                sol = batch.solve(obj_sparse_z, proj_z, step_size_z, z_init, line_search_sparse_z)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y))
                alphas[3].append(alpha)
                # plt.plot(x, log_y)
                # plt.show()

                # BB gradient in x
                sol = batch.solve_BB(obj, proj, line_search, x_init)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
                alphas[4].append(alpha)
                # plt.plot(x, log_y)

                # BB gradient in z
                sol = batch.solve_BB(obj_z, proj_z, line_search_z, z_init)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min_z)
                if abs(log_y[-1] - log_y[int(len(log_y)/4)]) <= 1 and ratio <= 0.3:
                    print 'bad initialization'
                else: alphas[5].append(alpha)
                # plt.plot(x, log_y)
                # plt.show()

                # BB sparse gradient in x
                sol = batch.solve_BB(obj_sparse, proj, line_search_sparse, x_init)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y))
                alphas[6].append(alpha)
                # plt.plot(x, log_y)

                # BB sparse gradient in z
                sol = batch.solve_BB(obj_sparse_z, proj_z, line_search_sparse_z, z_init)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y))
                if abs(log_y[-1] - log_y[int(len(log_y)/4)]) <= 1 and ratio <= 0.3:
                    print 'bad initialization'
                else: alphas[7].append(alpha)
                # plt.plot(x, log_y)
                # plt.show()

                # LBFGS gradient in x
                sol = batch.solve_LBFGS(obj, proj, line_search, x_init)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
                alphas[8].append(alpha)
                # plt.plot(x, log_y)

                # LBFGS gradient in z
                sol = batch.solve_LBFGS(obj_z, proj_z, line_search_z, z_init)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min_z)
                if abs(log_y[-1] - log_y[int(len(log_y)/4)]) <= 1 and ratio <= 0.3:
                    print 'bad initialization'
                else: alphas[9].append(alpha)
                # plt.plot(x, log_y)
                # plt.show()

                # LBFGS sparse gradient in x
                sol = batch.solve_LBFGS(obj_sparse, proj, line_search_sparse, x_init)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y))
                alphas[10].append(alpha)
                # plt.plot(x, log_y)

                # LBFGS sparse gradient in z
                sol = batch.solve_LBFGS(obj_sparse_z, proj_z, line_search_sparse_z, z_init)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y))
                if abs(log_y[-1] - log_y[int(len(log_y)/4)]) <= 1 and ratio <= 0.3:
                    print 'bad initialization'
                else: alphas[11].append(alpha)
                # plt.plot(x, log_y)
                # plt.show()

            dfs = []
            for i, name in enumerate(algorithms):
                num_trials = len(alphas[i])
                tuples = zip([distribution]*num_trials, [str(ratio)]*num_trials, 
                    [name]*num_trials, range(num_trials))
                index = pd.MultiIndex.from_tuples(tuples)
                dfs.append(pd.DataFrame(alphas[i], index=index, columns=['alpha']))

            # aggregate rates of convergence from specific measurement ratio
            dfs2.append(pd.concat(dfs))

        # aggregate rates of convergence data from specific distribution 
        results.append(pd.concat(dfs2))

        # aggregate coherence data from specific distribution 
        tuples = zip([distribution]*len(measurement_ratios), measurement_ratios)
        index = pd.MultiIndex.from_tuples(tuples)
        columns = ['coherence in x', 'avg product in x', 'coherence in z', 'avg coherence in z']
        coherences.append(pd.DataFrame(dfs3, index=index, columns=columns))

    results = pd.concat(results)
    coherences = pd.concat(coherences)
    #print results
    #print coherences
    results.save('results/rates_sparse.pkl')
    coherences.save('results/coherences_sparse.pkl')

if __name__ == '__main__':
    get_rate_of_convergence()



