import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

__author__ = 'jeromethai'

measurement_ratios = [0.01, 0.03, 0.1, 0.3, 1.0]
index = range(5)
distributions = ['uniform', 'aggregated']

algorithms = ['bb_x', 'bb_z', 'lbfgs_x', 'lbfgs_z']
#algorithms = ['bb_x_sparse', 'bb_z_sparse', 'lbfgs_x_sparse', 'lbfgs_z_sparse']
colors = ['--dk', '-db', '--og', '-or']

def display_progress():
    #coherence_data = pd.load('results/coherences_dense.pkl')
    rate_data = pd.load('results/rates_sparse.pkl')
    print rate_data
    for distribution in distributions:
        for i, algorithm in enumerate(algorithms):
            avg_rates = []
            for ratio in measurement_ratios:
                avg_rates.append(np.mean(rate_data.loc[distribution].loc[str(ratio)].loc[algorithm]))
            plt.plot(index, avg_rates, colors[i], linewidth=2, markersize=7, label=algorithm)
        plt.legend(loc=0)
        plt.title(distribution, fontsize=16)
        plt.xticks(range(5), [0.01,0.03,0.1,0.3,1])
        plt.xlabel('measurements / dimension', fontsize=16)
        plt.ylabel('log10 rate', fontsize=16)
        plt.show() 


if __name__ == '__main__':
    display_progress()