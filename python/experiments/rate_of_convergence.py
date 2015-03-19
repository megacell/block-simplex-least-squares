"""
In this experiment we aim at comparing the rate of convergence
between different gradient, bb, and lbfgs as the number of measurements
decreases
"""
import matplotlib.pyplot as plt
import pandas as pandas
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

experiment = 1
n = 1000
m2 = 1 # number of blocks
trials = 10
lasso = False
distribution = 'truncated'
in_z = False

dfs = []

def get_rate_of_convergence():
    for measurement_ratio in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]:
        m = int(measurement_ratio*n) # number of measurements

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
        print 'coherence in x', coherence(A)
        print 'coherence in z', coherence(Az)

        # get solver parts for which the rate where the rate alpha
        # for the step size has been optimized
        step_size, proj, line_search, obj = get_solver_parts((Q, c), 
                block_starts, 10., lasso=lasso)
        step_size_z, proj_z, line_search_z, obj_z = get_solver_parts((Qz, cz), 
                block_starts, 10., True, lasso=lasso)

        for i in range(trials):
            print 'trial', i+1

            # raw gradient in x
            x_init = np.random.rand(n)
            proj(x_init)
            sol = batch.solve(obj, proj, step_size, x_init, line_search)
            x, y = zip(*sol['progress'])
            x, log_y, alpha = clean_progress(np.array(x), np.array(y) - f_min)
            plt.plot(x, log_y)
            plt.show()





if __name__ == '__main__':
    get_rate_of_convergence()