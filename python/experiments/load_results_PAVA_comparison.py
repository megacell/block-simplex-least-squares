import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


__author__ = 'jeromethai'

dimensions = ['1e1', '1e2', '1e3', '1e4', '1e5', '1e6']
algorithms = ['sklearn', 'PAVA+', 'PAVA++']
index = range(6)


def display_progress():
    data = pd.load('results/PAVA_comparison_4.pkl')
    print data
    for algo in algorithms:
        avg_rate = []
        for n in dimensions:
            avg_rate.append(np.mean(data.loc[n][algo]))
        if algo == 'sklearn':
            plt.plot(index, avg_rate[:5]+[689.926681], label=algo, linewidth=2)
            plt.plot([0, 2, 4, 5], [avg_rate[0]/1.5, avg_rate[2]/14., avg_rate[4]/500., 689.926681/5000.], '--k', label='PAVA', linewidth=2)
        else:
            plt.plot(index, avg_rate, label=algo, linewidth=2)
        print avg_rate
    plt.legend(loc=0)
    plt.xticks(range(6), ['1e1','1e2','1e3','1e4','1e5','1e6'])
    plt.yscale('log')
    plt.ylabel('average time taken (s)', fontsize=20)
    plt.xlabel('problem size', fontsize=16)
    plt.show()



if __name__ == '__main__':
    display_progress()