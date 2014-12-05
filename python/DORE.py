import numpy as np
from numpy import linalg as la
import time
import logging

def solve(x0, linop, linop_T, target, record_every=5, proj=None,
        log=None, options=None, i = 10000, eps = 10**-16):
    """Solves DORE accelerated least squares via projection
    @param diagnostics is a closure that accepts the current solution and 
    iteration number
    """
    # Save initial state
    start = log(0,x0,0)

    # override defaults
    if options and 'max_iter' in options:
        i = options['max_iter']
    if options and 'opt_tol' in options:
        eps = options['opt_tol']

    b = -np.array(target)
    x = np.array(x0)
    n = x0.shape[0]

    x_start = x
    x_prev = x
    Ax = 0
    Ax_prev = 0
    Ax_prev_prev = 0

    for iter_ in xrange(i):
        Ax_prev_prev = Ax_prev
        Ax_prev = Ax
        Ax = linop(x)
        err = b - Ax
        norm_change = ((la.norm(x - x_prev)**2))

        if iter_ > 0 and (norm_change <= eps):
            break
        x_new = x + linop_T(err)

        x_new = proj(x_new)
        Ax = linop(x_new)
        err = b - Ax

        if iter_ > 2:
            delta_Ax = Ax - Ax_prev
            dp = delta_Ax.dot(delta_Ax)
            if dp > 0:
                a1 = delta_Ax.dot(err)/dp
                Ax_1 = (1+a1)*Ax - a1*Ax_prev
                x_1 = x_new + a1*(x_new - x)
                err_1 = b - Ax_1

                delta_Ax = Ax_1 - Ax_prev_prev
                dp = delta_Ax.dot(delta_Ax)
                if dp > 0:
                    a2 = delta_Ax.dot(err_1)/dp
                    x_2 = x_1 + a2*(x_1 - x_prev)
                    x_2 = proj(x_2)

                    Ax_2 = linop(x_2)
                    err_2 = b - Ax_2

                    if err_2.dot(err_2) / err.dot(err) < 1:
                        x_select = x_2
                        Ax = Ax_2
                    else:
                        x_select = x_new
                else:
                    x_select = x_new
            else:
                x_select = x_new
        else:
            x_select = x_new

        x_prev = x
        x = x_select

        # Save intermediate state
        if iter_ % record_every == 0:
            start = log(iter_,x,time.time()-start)
        if options and 'verbose' in options and options['verbose'] >= 1 and \
                iter_ % 100 == 0:
            logging.debug("iter=%d: %e %e %e" % (iter_,err.dot(err),norm_change,
                la.norm(x)))

    # Save final state
    log(iter_,x,time.time()-start)
    return x
