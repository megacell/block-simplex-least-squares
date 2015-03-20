"""
In this experiment we aim at comparing the rate of convergence
between different gradient, bb, and lbfgs as the number of measurements
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
                        random_least_squares, 
                        coherence,
                        ls_to_ls_in_z)
from data_utils import clean_progress

__author__ = 'jeromethai'

experiment = 5
algorithms = ['raw_x', 'raw_z', 'bb_x', 'bb_z', 'lbfgs_x', 'lbfgs_z']

n = 1000
m2 = 1 # number of blocks
trials = 10
lasso = False
in_z = False

measurement_ratios = [0.01, 0.05, 0.1, 0.5, 1.0]
distributions = ['truncated', 'exponential', 'normal', 
                    'cumulative_normal', 'log_normal']

def get_rate_of_convergence():

    results = []
    coherences = []

    for distribution in distributions:

        dfs2 = []
        dfs3 = []

        for ratio in measurement_ratios:
            m = int(ratio*n) # number of measurements

            # contruct blocks
            block_sizes = np.random.multinomial(n-m2,np.ones(m2)/m2) + np.ones(m2)
            assert sum(block_sizes) == n
            block_starts = np.append([0], np.cumsum(block_sizes[:-1])).astype(int)
            block_ends = np.append(block_starts[1:], [n])
            assert False not in ((block_ends-block_starts)>1)

            # generate random least squares
            Q, c, x_true, f_min, min_eig, A, b = random_least_squares(m, n, 
                    block_starts, 0.5, in_z=in_z, lasso=lasso, distribution=distribution)
            Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts, lasso=lasso)
            f_min_z = f_min - f0

            # get condition number and coherence
            Az = ls_to_ls_in_z(A, b, block_starts)[0]
            w = np.linalg.eig(Q)[0]
            wz = np.linalg.eig(Qz)[0]
            print 'condition number in x', w[0]/w[-1]
            print 'condition number in z', wz[0]/wz[-1]
            c_x = coherence(A)
            c_z = coherence(Az)
            dfs3.append([c_x[0], c_x[1], c_z[0], c_z[1]])
            print 'coherence in x', c_x[0], c_x[1]
            print 'coherence in z', c_z[0], c_z[1]

            # get solver parts for which the rate where the rate alpha
            # for the step size has been optimized
            step_size, proj, line_search, obj = get_solver_parts((Q, c), 
                    block_starts, 10., lasso=lasso)
            step_size_z, proj_z, line_search_z, obj_z = get_solver_parts((Qz, cz), 
                    block_starts, 10., True, lasso=lasso)

            alphas = [[] for i in range(6)]


            for i in range(trials):
                print 'trial', i+1

                # initialize
                x_init = np.random.rand(n)
                proj(x_init)
                z_init = x2z(x_init, block_starts=block_starts, lasso=lasso)

                # raw gradient in x
                sol = batch.solve(obj, proj, step_size, x_init, line_search)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
                alphas[0].append(alpha)
                #plt.plot(x, log_y)

                # raw gradient in z
                sol = batch.solve(obj_z, proj_z, step_size_z, z_init, line_search_z)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min_z)
                alphas[1].append(alpha)
                #plt.plot(x, log_y)
                #plt.show()

                # BB gradient in x
                sol = batch.solve_BB(obj, proj, line_search, x_init)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
                alphas[2].append(alpha)
                #plt.plot(x, log_y)

                # BB gradient in z
                sol = batch.solve_BB(obj_z, proj_z, line_search_z, z_init)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min_z)
                alphas[3].append(alpha)
                #plt.plot(x, log_y)
                #plt.show()

                # LBFGS in x
                sol = batch.solve_LBFGS(obj, proj, line_search, x_init)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
                alphas[4].append(alpha)
                #plt.plot(x, log_y)

                # LBFGS in z
                sol = batch.solve_LBFGS(obj_z, proj_z, line_search_z, z_init)
                x, y = zip(*sol['progress'])
                x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min_z)
                alphas[5].append(alpha)
                #plt.plot(x, log_y)
                #plt.show()

            dfs = []
            for i, name in enumerate(algorithms):
                tuples = zip([distribution]*trials, [ratio]*trials, 
                    [name]*trials, range(trials))
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
    print results
    print coherences


if __name__ == '__main__':
    get_rate_of_convergence()