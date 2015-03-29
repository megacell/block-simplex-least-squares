import time
import sys
sys.path.append('../')
from c_extensions.c_extensions import (isotonic_regression_c,
                                    isotonic_regression_c_2,
                                    isotonic_regression_c_3)
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


def worst_case():

    dimensions = [int(1e1), int(1e2), int(1e3), int(1e4), int(1e5), int(1e6)]
    trials = 5
    times = []
    rs = check_random_state(0)

    for n in dimensions:
        print 'dimension', n
        for trial in range(trials):
            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
            
            # in-place PAVA on log data
            start_time = time.time()
            isotonic_regression_c_2(np.copy(y), 0, n)
            time_log = time.time() - start_time

            if n <= int(1e5):
                y = np.arange(n).astype(float)
                y[-1] = -1e12
                # in-place PAVA worst case
                start_time = time.time()
                isotonic_regression_c_2(np.copy(y), 0, n)
                time_worst = time.time() - start_time
            else:
                time_worst = -1.

            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
            # in-place PAVA on log data
            start_time = time.time()
            isotonic_regression_c(np.copy(y), 0, n)
            time_log_2 = time.time() - start_time

            if n <= int(1e5):
                y = np.arange(n).astype(float)
                y[-1] = -1e12
                # in-place PAVA worst case
                start_time = time.time()
                isotonic_regression_c(np.copy(y), 0, n)
                time_worst_2 = time.time() - start_time
            else:
                time_worst_2 = -1.

            times.append([time_log, time_worst, time_log_2, time_worst_2])

    #print times
    index = []
    for n in ['1e1','1e2','1e3','1e4','1e5', '1e6']: index += [n]*trials
    columns = ['log_data_PAVA+', 'worst_case_PAVA+', 'log_data_PAVA++', 'worst_case_PAVA++']
    df = pd.DataFrame(times, index=index, columns=columns)
    print df
    df.save('results/PAVA_worst_case.pkl')

if __name__ == '__main__':
    worst_case()
