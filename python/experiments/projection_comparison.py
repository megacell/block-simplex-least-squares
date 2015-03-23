import time

__author__ = 'jeromethai'

import sys
sys.path.append('../')
from c_extensions.c_extensions import (isotonic_regression_c,
                                              isotonic_regression_c_2,
                                              isotonic_regression_sparse_c,
                                              proj_simplex_c)
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
import pandas as pd
from bsls_utils import block_sizes_to_N


def compare_PAVA_implementations():
    trials = 10
    rs = check_random_state(0)
    times = []
    dimensions = [int(1e1), int(1e2), int(1e3), int(1e4), int(1e5), int(1e6)]
    #dimensions = [int(1e6)]

    for n in dimensions:
        print 'dimensionality', n
        x = np.arange(n)
        for trial in range(trials):

            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))

            # scikit-learn PAVA
            if n <= int(1e5):
            #if n <= int(1e6):
                ir = IsotonicRegression()
                y_copy = np.copy(y)
                start_time = time.time()
                ir.fit_transform(x, y_copy)
                time1 = time.time() - start_time
            else: time1 = -1.

            # in-place PAVA
            y_copy = np.copy(y)
            start_time = time.time()
            isotonic_regression_c_2(y_copy, 0, n)
            time2 = time.time() - start_time

            # in-place PAVA++
            y_copy = np.copy(y)
            start_time = time.time()
            isotonic_regression_c(y_copy, 0, n)
            time3 = time.time() - start_time

            times.append([time1, time2, time3])

    index = []
    for n in ['1e1','1e2','1e3','1e4','1e5','1e6']: index += [n]*trials
    #for n in ['1e6']: index += [n]*trials  
    tuples = zip()
    df = pd.DataFrame(times, index=index, columns=['sklearn', 'PAVA+', 'PAVA++'])
    print df
    df.save('results/PAVA_comparison_5.pkl')


def compare_PAVA_sparse():
    trials = 5
    times = []
    rs = check_random_state(0)
    n = int(1e6)
    densities = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    for density in densities:
        print 'sparsity', 1-density

        for trial in range(trials):

            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
            # choose int(density*n) elements at random 
            indices = np.sort(np.random.choice(n, int(density*n), replace=False))
            starts = np.append([0], indices)
            ends = np.append(indices, [n])
            weights = np.ones(n).astype(int)
            w = np.copy(weights)
            for s,e in zip(starts,ends):
                y[s:e] = y[s]*np.ones(e-s)
                weights[s] = e-s
            #print zip(y, weights)

            # in-place PAVA
            y_copy = np.copy(y)
            start_time = time.time()
            isotonic_regression_c_2(y_copy, 0, n)
            time1 = time.time() - start_time

            # in-place PAVA++
            y_copy = np.copy(y)
            start_time = time.time()
            isotonic_regression_sparse_c(y_copy, 0, n, w)
            time2 = time.time() - start_time

            # sparse in-place PAVA
            y_copy = np.copy(y)
            start_time = time.time()
            isotonic_regression_sparse_c(y_copy, 0, n, weights)
            time3 = time.time() - start_time

            times.append([time1, time2, time3])

    index = []
    for n in ['0.3', '0.1', '0.03', '0.01', '0.003', '0.001']: index += [n]*trials
    #for n in ['1e6']: index += [n]*trials  
    tuples = zip()
    df = pd.DataFrame(times, index=index, columns=['PAVA+', 'PAVA++', 'PAVA_sparse'])
    df.save('results/PAVA_sparse_comparison.pkl')
    print df


def comparison_PAVA_simplex_proj():
    
    dimensions = [int(1e1), int(1e2), int(1e3), int(1e4), int(1e5), int(1e6), int(1e7)]
    dimensions = [int(1e1), int(1e2), int(1e3), int(1e4), int(1e5), int(1e6)]
    trials = 1
    rs = check_random_state(0)

    for n in dimensions:
        print 'dimensionality', n
        N = block_sizes_to_N([n+1])

        for trial in range(trials):
            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))

            # in-place PAVA++
            y_copy = np.copy(y)
            start_time = time.time()
            isotonic_regression_c(y, 0, n)
            time1 = time.time() - start_time

            # simplex-projection
            w = N*y
            w[-1] += 1
            start_time = time.time()
            proj_simplex_c(w, 0, n+1)
            time2 = time.time() - start_time

            print time1, time2

if __name__ == '__main__':
    #compare_PAVA_implementations()
    compare_PAVA_sparse()
    #comparison_PAVA_simplex_proj()

