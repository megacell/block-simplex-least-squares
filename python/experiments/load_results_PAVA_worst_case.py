import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


__author__ = 'jeromethai'

dimensions = ['1e1', '1e2', '1e3', '1e4', '1e5', '1e6']
scenarios = ['log_data_PAVA+', 'worst_case_PAVA+', 'log_data_PAVA++', 'worst_case_PAVA++']
index = range(6)

def display_progress():
    data = pd.load('results/PAVA_worst_case.pkl')
    print data
    for scenario in scenarios:
        avg_rate = []
        for n in dimensions:
            avg_rate.append(np.mean(data.loc[n][scenario]))
        if scenario == 'worst_case_PAVA+' or scenario == 'worst_case_PAVA++':
            plt.plot(index[:-1], avg_rate[:-1], label=scenario, linewidth=2)
        else:
            plt.plot(index, avg_rate, label=scenario, linewidth=2)
    plt.legend(loc=0)
    plt.xticks(range(6), ['1e1','1e2','1e3','1e4','1e5','1e6'])
    plt.yscale('log')
    plt.ylabel('average time taken (s)', fontsize=20)
    plt.xlabel('problem size', fontsize=16)
    plt.title('Timings of in-place PAVA', fontsize=20)
    plt.show()



if __name__ == '__main__':
    display_progress()