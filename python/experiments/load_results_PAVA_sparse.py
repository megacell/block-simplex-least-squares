import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


__author__ = 'jeromethai'


algorithms = ['PAVA+', 'PAVA++', 'PAVA_sparse']
data = pd.load('results/PAVA_sparse_comparison.pkl')
densities = ['0.3', '0.1', '0.03', '0.01', '0.003', '0.001']
index = range(6)


def display_progress():
    for algo in algorithms:
        avg_rate = []
        for n in densities:
            avg_rate.append(np.mean(data.loc[n][algo]))
        if algo == 'PAVA_sparse':
            plt.plot(index, avg_rate, label='PAVA++ with boost', linewidth=2)
        else:
            plt.plot(index, avg_rate, label=algo, linewidth=2)
        print avg_rate
    plt.legend(loc=0)
    plt.xticks(range(6), ['0.3', '0.1', '0.03', '0.01', '0.003', '0.001'])
    plt.yscale('log')
    plt.ylabel('average time taken (s)', fontsize=20)
    plt.xlabel('non-zeros / dimension', fontsize=16)
    plt.show()

if __name__ == '__main__':
    display_progress()