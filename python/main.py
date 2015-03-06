import scipy.io as sio
import matplotlib.pyplot as plt
import argparse
import logging
import numpy as np
import numpy.linalg as la

import solvers
import util
from proj_simplex.simplex_projection import simplex_projection
# from projection import pysimplex_projection
import BB, LBFGS, DORE
import config as c
from util import load_data

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='Data file (*.mat)',
                        default='route_assignment_matrices_ntt.mat')
    parser.add_argument('--log', dest='log', nargs='?', const='INFO',
            default='WARN', help='Set log level (default: WARN)')
    parser.add_argument('--method',dest='method',type=str,default='BB',
                        help='Least squares method')
    parser.add_argument('--init',dest='init',action='store_true',
                        default=False,help='Initial solution from data')
    parser.add_argument('--eq',dest='eq',type=str,default='CP',
                        help='Type of equality constraint (CP or OD)')
    parser.add_argument('--noise',dest='noise',type=float,default=None,
            help='Noise level')
    return parser

def LS_solve(A,b,x0,N,block_sizes,method):
    z0 = util.x2z(x0,block_sizes)
    target = A.dot(x0)-b

    options = { 'max_iter': 300000,
                'verbose': 1,
                'opt_tol' : 1e-30,
                'suff_dec': 0.003, # FIXME unused
                'corrections': 500 } # FIXME unused
    AT = A.T.tocsr()
    NT = N.T.tocsr()

    f = lambda z: 0.5 * la.norm(A.dot(N.dot(z)) + target)**2
    nabla_f = lambda z: NT.dot(AT.dot(A.dot(N.dot(z)) + target))

    def proj(x):
        projected_value = simplex_projection(block_sizes - 1,x)
        # projected_value = pysimplex_projection(block_sizes - 1,x)
        return projected_value

    import time
    iters, times, states = [], [], []
    def log(iter_,state,duration):
        iters.append(iter_)
        times.append(duration)
        states.append(state)
        start = time.time()
        return start

    logging.debug('Starting %s solver...' % method)
    if method == 'LBFGS':
        LBFGS.solve(z0+1, f, nabla_f, solvers.stopping, log=log,proj=proj,
                    options=options)
        logging.debug("Took %s time" % str(np.sum(times)))
    elif method == 'BB':
        BB.solve(z0,f,nabla_f,solvers.stopping,log=log,proj=proj,
                 options=options)
    elif method == 'DORE':
        # setup for DORE
        alpha = 0.99
        lsv = util.lsv_operator(A, N)
        logging.info("Largest singular value: %s" % lsv)
        A_dore = A*alpha/lsv
        target_dore = target*alpha/lsv

        DORE.solve(z0, lambda z: A_dore.dot(N.dot(z)),
                   lambda b: N.T.dot(A_dore.T.dot(b)), target_dore, proj=proj,
                   log=log,options=options,record_every=100)
        A_dore = None
    logging.debug('Stopping %s solver...' % method)
    return iters, times, states

def LS_postprocess(states, x0, A, b, x_true, scaling=None, block_sizes=None,
                   output=None, N=None, is_x=False):
    if x_true is None:
        return [], [], output
    if scaling is None:
        scaling = np.ones(x_true.shape)
    if output is None:
        output = {}
    d = len(states)

    # Convert back to x (from z) if necessary
    if not is_x and N.size > 0:
        x_hat = N.dot(np.array(states).T) + np.tile(x0,(d,1)).T
    else:
        x_hat = np.array(states).T
    x_last = x_hat[:,-1]
    n = x_hat.shape[1]

    # Record sizes
    output['AA'] = A.shape
    output['x_hat'] = x_hat.shape
    output['blocks'] = block_sizes.shape if block_sizes is not None else None

    # Objective error, i.e. 0.5||Ax-b||_2^2
    starting_error = 0.5 * la.norm(A.dot(x0)-b)**2
    opt_error = 0.5 * la.norm(A.dot(x_true)-b)**2
    diff = A.dot(x_hat) - np.tile(b,(d,1)).T
    error = 0.5 * np.diag(diff.T.dot(diff))
    output['0.5norm(Ax-b)^2'], output['0.5norm(Ax_init-b)^2'] = error, starting_error
    output['0.5norm(Ax*-b)^2'] = opt_error

    # Route flow error, i.e ||x-x*||_1
    x_true_block = np.tile(x_true,(n,1))
    x_diff = x_true_block-x_hat.T

    scaling_block = np.tile(scaling,(n,1))
    x_diff_scaled = scaling_block * x_diff
    x_true_scaled = scaling_block * x_true_block

    # most incorrect entry (route flow)
    dist_from_true = np.max(x_diff_scaled,axis=1)
    output['max|f * (x-x_true)|'] = dist_from_true

    # num incorrect entries
    wrong = np.bincount(np.where(x_diff > 1e-3)[0])
    output['incorrect x entries'] = wrong

    # % route flow error
    per_flow = np.sum(np.abs(x_diff_scaled), axis=1) / np.sum(x_true_scaled, axis=1)
    output['percent flow allocated incorrectly'] = per_flow

    # initial route flow error
    start_dist_from_true = np.max(scaling * np.abs(x_true-x0))
    output['max|f * (x_init-x_true)|'] = start_dist_from_true

    return x_last, error, output

def LS_plot(x_last, times, error):
    plt.figure()
    plt.hist(x_last)

    plt.figure()
    plt.loglog(np.cumsum(times),error)
    plt.show()

def main(args=None,plot=False):
    if args is None:
        p = parser()
        args = p.parse_args()
    if args.log in c.ACCEPTED_LOG_LEVELS:
        logging.basicConfig(level=eval('logging.'+args.log))

    # load data
    output=None

    A, b, N, block_sizes, x_true, nz, flow, rsort_index, x0, out = \
        load_data(args.file, eq=args.eq, init=args.init)

    if args.noise:
        delta = np.random.normal(scale=b*args.noise)
        b = b + delta

    iters, times, states = LS_solve(A,b,x0,N,block_sizes,args.method)
    x_last, error, output = LS_postprocess(states,x0,A,b,
                                                x_true,scaling=flow,
                                                block_sizes=block_sizes,N=N,
                                                output=output)
    if plot:
        LS_plot(x_last, times, error)

    return iters, times, states, output

if __name__ == "__main__":
    iters, times, states, output = main()
