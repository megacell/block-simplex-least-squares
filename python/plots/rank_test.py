import scipy.sparse as sps
import scipy.sparse.linalg as sla
import scipy.io as sio
import numpy as np

import pickle as pkl

import util2 as util

from scipy.sparse.linalg.interface import aslinearoperator, LinearOperator
from scipy.sparse.linalg.eigen.arpack import eigsh

import matplotlib.pyplot as plt

def compute_rank_approx(sz, routes):
    A, b, N, block_sizes, x_true = util.load_data(str(sz)+"/experiment2_waypoints_matrices_routes_"+str(routes)+".mat")
    def matvec_XH_X(x):                                                                 
        return A.dot(A.T.dot(x))
    XH_X = LinearOperator(matvec=matvec_XH_X, dtype=A.dtype, shape=(A.shape[0], A.shape[0]))
    eigvals, eigvec = eigsh(XH_X, k=500, tol=10**-5)
    eigvals = eigvals[::-1]
    for i, val in enumerate(eigvals):
        if val < 10**-6:
            return (N.shape[1], i)

route_densities = [3, 10, 20, 30, 40, 50]
colors = ['m', 'c', 'b', 'k', 'g', 'r']

plt.hold(True)

for route_density, color in zip(route_densities, colors):
    num_of_pp = sorted([3325, 2375, 2850, 475, 950, 1425, 1900, 238, 3800, 713])
    null_sizes = []
    for pp_cnt in sorted(num_of_pp):
        size_z, i = compute_rank_approx(pp_cnt, route_density)
        print (str(pp_cnt)+":"), size_z - i
        null_sizes.append(size_z - i)
    
    print color
    plt.plot(num_of_pp, null_sizes, '-o'+color, label=str(route_density)+' Routes')
    plt.plot([0, 3800], [OD_only, OD_only], '-'+color)

plt.title('Degree of freedom in MATSim from cell + OD data (PM)', fontweight='bold')
plt.xlabel('Cells')
plt.yscale('log')
plt.ylabel('Degrees of freedom')
plt.legend()
plt.savefig('degrees_of_freedom_pm.svg')
plt.savefig('degrees_of_freedom_pm.png')
plt.clf()
