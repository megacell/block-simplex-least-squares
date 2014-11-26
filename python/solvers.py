from util import is_ones, all_equal, block_e, PROB_SIMPLEX, EQ_CONSTR_ELIM, \
        L_BFGS, ADMM, SPG
import numpy as np
import numpy.linalg as la
import logging
import scipy.io as sio
from numpy import ones, array

def qp(P, q, G=None, h=None, A=None, b=None, solver=None, initvals=None):
    from cvxopt.solvers import qp
    return qp(P, q, G, h, A, b, solver, initvals)

def qp2(P, q, G=None, h=None, A=None, b=None, solver=None, initvals=None, \
        N=None, block_sizes=None, reduction=None, constraints=None, method=None):

    if block_sizes != None and PROB_SIMPLEX in constraints:
        if reduction == EQ_CONSTR_ELIM: # work in z's
            if method == L_BFGS:
                pass
            elif method == SPG:
                return spg.solver(P, q, G, h, A, b, solver, initvals, N, block_sizes, reduction, constraints)
            elif method == ADMM:
                pass
        else: # work in x's
            x_0 = block_e(block_sizes-1,block_sizes)
            print x_0
            if method == L_BFGS:
                pass
            elif method == SPG:
                pass
            elif method == ADMM:
                pass
    else:
        print "'block_sizes' parameter required"

def least_squares(x, linop, linop_transpose, target, proj=None, diagnostics=None, options=None):
    import DORE
    return DORE.solve(x, linop, linop_transpose, target, proj=proj, diagnostics=diagnostics,options=options)

# Stopping condition
def stopping(g,fx,i,t,d=None,delta_g=None,options=None,TOLER=1e-6):
    if options and 'max_iter' in options:
        if i >= options['max_iter']:
            return True
    if options and 'opt_tol' in options:
        TOLER = options['opt_tol']

    norm2_nabla_f = np.square(la.norm(g))
    thresh = TOLER * (1 + abs(fx))
    if options and 'verbose' in options and options['verbose'] >= 1 and i % 20 == 0:
        logging.debug("iter=%d: %e %e %e %f" % (i,t,norm2_nabla_f,thresh,fx))
    if norm2_nabla_f <= thresh:
        logging.info("iter=%d: %e %e %e %f" % (i,t,norm2_nabla_f,thresh,fx))
        logging.warning('Exiting... norm(grad) too small')
        return True
    if type(d) != type(None) and la.norm(t*d) <= 1e-12:
        logging.info("iter=%d: %e %e %e %f" % (i,t,norm2_nabla_f,thresh,fx))
        logging.warning('Exiting... step too small')
        return True
    if type(delta_g) != type(None) and la.norm(delta_g) == 0:
        logging.info("iter=%d: %e %e %e %f" % (i,t,norm2_nabla_f,thresh,fx))
        logging.warning('Exiting... no change in gradient')
        return True
    return False
