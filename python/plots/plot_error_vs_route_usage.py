import python.config as config
import matplotlib.pyplot as pyplot
import scipy.io as sio
import scipy
import numpy as np
import python.util as util
import os
from scipy.sparse.linalg import LinearOperator
import scipy.linalg.interpolative as inter
import pickle
BASE_DIR = '%s/%s'%(config.DATA_DIR, config.EXPERIMENT_MATRICES_DIR)
b_50 = None
def b_estimate():
    global b_50
    if b_50 == None:
         A, b_50, N, block_sizes, x_true, nz, flow = util.load_data('%s/experiment2_control_matrices_routes_%s.mat'% (BASE_DIR, 2000))
    return b_50
def read_problem_matrices(filepath):
    A, b, N, block_sizes, x_true, nz, flow = util.load_data(filepath)
    return A, x_true, b_estimate(), N, block_sizes, flow

def read_x_computed(filepath, block_sizes, N):
    matrices = sio.loadmat(filepath)
    x0 = np.array([util.block_e(block_sizes - 1, block_sizes)])
    x = matrices['x']
    fx = matrices['fx']
    x = x0.T + N*x.T
    return x, fx

def metrics(A,b,X):
    d = X.shape[1]
    diff = A.dot(X) - np.tile(b,(d,1)).T
    error = 0.5 * np.diag(diff.T.dot(diff))
    RMSE = np.sqrt(error/b.size)
    den = np.sum(b)/np.sqrt(b.size)
    pRMSE = RMSE / den
    plus = A.dot(X) + np.tile(b,(d,1)).T

    # GEH metric [See https://en.wikipedia.org/wiki/GEH_statistic]
    GEH = np.sqrt(2 * diff**2 / np.abs(plus))
    meanGEH = np.mean(GEH,axis=0)
    maxGEH = np.max(GEH,axis=0)
    GEHunder5 = np.mean(GEH < 5,axis=0)
    GEHunder1 = np.mean(GEH < 1,axis=0)
    GEHunder05 = np.mean(GEH < 0.5,axis=0)
    GEHunder005 = np.mean(GEH < 0.05,axis=0)
    return { 'error': error, 'RMSE': RMSE, 'pRMSE': pRMSE,
        'mean_GEH': meanGEH, 'max_GEH': maxGEH,
        'GEH_under_5': GEHunder5,'GEH_under_1': GEHunder1,
        'GEH_under_0.5': GEHunder05,'GEH_under_0.05': GEHunder005
        }

def flowerror(x_sol, x_true, flow):
    return np.sum(flow * np.abs((x_sol-np.matrix(x_true).T))) / np.sum(flow * x_true)

def flowerror_larger_than(x_sol, x_true, flow, n):
    gtn = flow > n
    x_sol=x_sol[gtn]
    x_true = x_true[gtn]
    flow = flow[gtn]
    return np.sum(flow * np.abs((x_sol-np.matrix(x_true).T))) / np.sum(flow * x_true)

def get_statistics_from(solution_file, problem_file):
    print problem_file, solution_file
    A, x, b, N, block_size, flow= read_problem_matrices(problem_file)

    #print(A.shape)
    #print N.shape
    x_hat, fx = read_x_computed(solution_file,block_size, N)
    geh = metrics(A, b, x_hat)
    print flowerror(x_hat, x, flow)
    geh['flow_per_error'] = flowerror(x_hat, x, flow)
    return geh

def plot_GEH(routes, GEH, errorstat, xlabel, ylabel):
    GEH_mean = [g[errorstat] for g in GEH]
    pyplot.plot(routes, GEH_mean, '-o')
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)


def matrix_files(directory):
    routes = [3, 10, 20, 30, 40, 50]
    matrixfiles = ['%s/experiment2_waypoints_matrices_routes_%s.mat' % (directory, i) for i in routes]
    return matrixfiles


def plot_GEH_vs_route_number_waypoints(directory, save_directory):
    routes = [3, 10, 20, 30, 40, 50]
    outputfiles = ['%s/output_waypoints%s.mat'% (directory, i) for i in routes]
    matrixfiles = matrix_files(directory)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    statistics = [get_statistics_from(o, m) for o, m in zip(outputfiles, matrixfiles)]
    # plot_GEH(routes, statistics, 'mean_GEH', "Maximum Route Number", "Mean GEH")
    # pyplot.savefig("{0}/{1}.{2}".format(save_directory, 'mean_GEH', "png"))
    # pyplot.figure()
    # plot_GEH(routes, statistics, 'GEH_under_5', "Maximum Route Number", "Percent GEH under 5")
    # pyplot.savefig("{0}/{1}.{2}".format(save_directory, 'GEH_under_5', "png"))
    # pyplot.figure()
    # plot_GEH(routes, statistics, 'GEH_under_0.5', "Maximum Route Number", "Percent GEH under .5")
    # pyplot.savefig("{0}/{1}.{2}".format(save_directory, 'GEH_under_0.5', "png"))
    # pyplot.figure()
    # plot_GEH(routes, statistics, 'GEH_under_0.05', "Maximum Route Number", "Percent GEH under .05")
    # pyplot.savefig("{0}/{1}.{2}".format(save_directory, 'GEH_under_0.05', "png"))
    # pyplot.figure()
    # plot_GEH(routes, statistics, 'flow_per_error', "Maximum Route Number", "Percent Flow Allocated Incorrectly")
    # pyplot.savefig("{0}/{1}.{2}".format(save_directory, 'flow_per_error', "png"))
    # pyplot.figure()
    # pyplot.close('all')
    return statistics

def read_ranks(density):
    mf = matrix_files("{0}/{1}".format(BASE_DIR,density))
    def readAU(f):
        data = sio.loadmat(f)
        A = data['A']
        U = data['U']
        return A, U
    def rank(A,U):
        stacked = scipy.sparse.vstack((A, U))
        print scipy.sparse.issparse(stacked)
        return scipy.linalg.interpolative.estimate_rank(scipy.sparse.linalg.aslinearoperator(stacked),.1)

    return [rank(A,U) for A, U in [readAU(f) for f in mf]]
if __name__== '__main__':
    density = config.WAYPOINT_DENSITIES
    stats = {d:plot_GEH_vs_route_number_waypoints("{0}/{1}".format(BASE_DIR,d), "{0}/{1}".format(config.DATA_DIR, d)) for d in density}
    pickle.dump(stats, open(config.PLOT_DIR +'/stats.pkl','w'))
    #read_ranks(3800)
    #print(read_ranks(density[0]))
