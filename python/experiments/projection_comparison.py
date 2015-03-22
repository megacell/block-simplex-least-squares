import time

__author__ = 'jeromethai'

import sys
sys.path.append('../../')
from python.c_extensions.c_extensions import (isotonic_regression_c,
                                              isotonic_regression_multi_c,
                                              isotonic_regression_c_2,
                                              isotonic_regression_multi_c_2)
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
import pandas as pd

def compare_PAVA_implementations():
    trials = 10
    rs = check_random_state(0)
    times = []

    for n in [int(1e1), int(1e2), int(1e3), int(1e4), int(1e5), int(1e6)]:
        print 'dimensionality', n
        x = np.arange(n)
        for trial in range(trials):

            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))

            # scikit-learn PAVA
            if n <= int(1e5):
                ir = IsotonicRegression()
                y_copy = np.copy(y)
                start_time = time.time()
                ir.fit_transform(x, y_copy)
                time1 = time.time() - start_time
            else: time1 = -1.

            # in-place PAVA
            y_copy = np.copy(y)
            start_time = time.time()
            isotonic_regression_c_2(y, 0, n)
            time2 = time.time() - start_time

            # in-place PAVA++
            y_copy = np.copy(y)
            start_time = time.time()
            isotonic_regression_c(y, 0, n)
            time3 = time.time() - start_time

            times.append([time1, time2, time3])

    index = []
    for n in ['1e1','1e2','1e3','1e4','1e5','1e6']: index += [n]*trials 
    tuples = zip()
    df = pd.DataFrame(times, index=index, columns=['sklearn', 'PAVA+', 'PAVA++'])
    print df
    df.save('results/PAVA_comparison.pkl')


def comparison_PAVA_simplex_proj():
    pass



if __name__ == '__main__':
    #compare_PAVA_implementations()
    comparison_PAVA_simplex_proj()