import pickle
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == '__main__':
    progress = pd.load('progress.pkl')

    plt.plot(progress.loc['bb_x_2']['time'], progress.loc['bb_x_2']['f-f_min'], label='x')
    plt.plot(progress.loc['bb_z_2']['time'], progress.loc['bb_z_2']['f-f_min'], label='z')
    plt.yscale('log')
    plt.legend(loc=0)
    plt.show()

    plt.plot(progress.loc['lbfgs_x_0']['time'], progress.loc['lbfgs_x_0']['f-f_min'], label='x')
    plt.plot(progress.loc['lbfgs_z_0']['time'], progress.loc['lbfgs_z_0']['f-f_min'], label='z')
    plt.yscale('log')
    plt.legend(loc=0)
    plt.show()

    plt.plot(progress.loc['lbfgs_x_1']['time'], progress.loc['lbfgs_x_1']['f-f_min'], label='x')
    plt.plot(progress.loc['lbfgs_z_1']['time'], progress.loc['lbfgs_z_1']['f-f_min'], label='z')
    plt.yscale('log')
    plt.legend(loc=0)
    plt.show()

    plt.plot(progress.loc['lbfgs_x_2']['time'], progress.loc['lbfgs_x_2']['f-f_min'], label='x')
    plt.plot(progress.loc['lbfgs_z_2']['time'], progress.loc['lbfgs_z_2']['f-f_min'], label='z')
    plt.yscale('log')
    plt.legend(loc=0)
    plt.show()


    # plt.plot(index, est_lf[0], '-or', label='With OD flows')
    # for i in range(D):
    #     plt.plot(index, est_wp[i], '-o'+color[i], label='With {} cells'.format(num_wps[i]))
    # plt.title('Path flow errors for network in ' + mode + ': OD vs ' + string)
    # plt.xlabel('Percentage of links observed (%)')
    # plt.ylabel('Relative error')
    # plt.yscale('log')
    # plt.legend(loc=0)
    # plt.show()