import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def display_progress():

    progress = pd.load('results/progress.pkl')

    for algo in ['batch', 'bb', 'lbfgs']:
        for i in range(1):
            
            x = np.array(progress.loc[algo+'_x_'+str(i)]['time'])
            log_y = np.log10(np.array(progress.loc[algo+'_x_'+str(i)]['f-f_min']))

            alpha = x.T.dot(log_y-log_y[0]) / x.T.dot(x)
            plt.plot(x, log_y, 'r', label='x')
            plt.plot(x, alpha*x + log_y[0], '--r', label='x')

            x = np.array(progress.loc[algo+'_z_'+str(i)]['time'])
            log_y = np.log10(np.array(progress.loc[algo+'_z_'+str(i)]['f-f_min']))

            alpha = x.T.dot(log_y-log_y[0]) / x.T.dot(x)
            plt.plot(x, log_y, 'g', label='z')
            plt.plot(x, alpha*x + log_y[0], '--g', label='x')

            plt.legend(loc=0)
            plt.title(algo+' experiment '+str(i))
            plt.show()

def display_progress_sparse():

    progress = pd.load('results/progress_sparse.pkl')

    for algo in ['batch', 'bb', 'lbfgs']:
        for i in range(1):

            x = progress.loc[algo+'_x_dense_'+str(i)]['time']
            y = progress.loc[algo+'_x_dense_'+str(i)]['f-f_min']
            plt.plot(x, y, 'r', label='x dense')

            x = progress.loc[algo+'_z_dense_'+str(i)]['time']
            y = progress.loc[algo+'_z_dense_'+str(i)]['f-f_min']
            plt.plot(x, y, 'g', label='z dense')

            x = progress.loc[algo+'_x_sparse_'+str(i)]['time']
            y = progress.loc[algo+'_x_sparse_'+str(i)]['f-f_min']
            plt.plot(x, y, '--r', label='x_sparse')

            x = progress.loc[algo+'_z_sparse_'+str(i)]['time']
            y = progress.loc[algo+'_z_sparse_'+str(i)]['f-f_min']            
            plt.plot(x, y, '--g', label='z_sparse')
            #plt.xscale('log')
            plt.yscale('log')
            plt.legend(loc=0)
            #plt.title(algo+' experiment '+str(i))
            plt.show()


if __name__ == '__main__':

    display_progress()
    #display_progress_sparse()

    # plt.plot(progress.loc['bb_x_2']['time'], progress.loc['bb_x_2']['f-f_min'], label='x')
    # plt.plot(progress.loc['bb_z_2']['time'], progress.loc['bb_z_2']['f-f_min'], label='z')
    # plt.yscale('log')
    # plt.legend(loc=0)
    # plt.title('BB ')
    # plt.show()



    # plt.plot(index, est_lf[0], '-or', label='With OD flows')
    # for i in range(D):
    #     plt.plot(index, est_wp[i], '-o'+color[i], label='With {} cells'.format(num_wps[i]))
    # plt.title('Path flow errors for network in ' + mode + ': OD vs ' + string)
    # plt.xlabel('Percentage of links observed (%)')
    # plt.ylabel('Relative error')
    # plt.yscale('log')
    # plt.legend(loc=0)
    # plt.show()