import numpy as np
import time
from algorithm_utils import stopping, normalization
from collections import deque


def solve(obj, proj, step_size, x_init, line_search=None, f_min=None, opt_tol=1e-6, 
          max_iter=1000, prog_tol=1e-9):
    """Projected batch gradient descent with line search
    obj: f,g = obj(x) with f the objective value and g the gradient at x
    proj: w = proj(x)
    step_size: step_size
    x_init: initial point
    f_min: optimal objective value
    opt_tol: stop when f-f_min < opt_tol
    max_iter: stop when iter == max_iter 
    prog_tol: stop when progression in f_old-f < prog_tol
    """
    t_proj = 0.
    t_obj = 0.
    t_line = 0.
    # allocate memories and initialize
    n = x_init.shape[0]
    x = x_init
    g = np.zeros(n)
    g_new = np.zeros(n)
    x_new = np.zeros(n)
    f_old = np.inf
    i = 1
    f = obj(x, g) # should update content of g
    progress = []
    start_time = time.time()
    while True:
        flag, stop = stopping(i, max_iter, f, f_old, opt_tol, prog_tol, f_min)
        if flag is True: break
        # update and project x
        t = step_size(i)
        np.add(x, -t*g, x_new) # should update content of x_new
        proj(x_new)
        f_new = obj(x_new, g_new) # should update content of g_new
        # do line search between x and x_new, should update x_new, g_new
        if line_search is not None:
            f_new = line_search(x, f, g, x_new, f_new, g_new, i)
        # take step
        f_old = f
        f = f_new
        np.copyto(x, x_new)
        np.copyto(g, g_new)
        i += 1
        progress.append([time.time() - start_time, f])
    return {'f': f, 'x': x, 'stop': stop, 'iterations': i, 
            'progress': progress}


def solve_BB(obj, proj, line_search, x_init, f_min=None, opt_tol=1e-6, 
          max_iter=1000, prog_tol=1e-9):
    """Projected batch gradient descent with Barzilei-Bornwein step 
    obj: f,g = obj(x) with f the objective value and g the gradient at x
    proj: w = proj(x)
    line_search: customized line search for the obj function
    x_init: initial point
    f_min: optimal objective value
    opt_tol: stop when f-f_min < opt_tol
    max_iter: stop when iter == max_iter 
    prog_tol: stop when progression in f_old-f < prog_tol
    """
    t_proj = 0.
    t_obj = 0.
    t_line = 0.
    # allocate memories and initialize
    n = x_init.shape[0]
    x = x_init
    g = np.zeros(n)
    delta_x = np.zeros(n)
    delta_g = np.zeros(n)
    g_new = np.zeros(n)
    x_new = np.zeros(n)
    f_old = np.inf
    i = 1
    f = obj(x, g) # should update content of g
    progress = []
    start_time = time.time()
    while True:
        flag, stop = stopping(i, max_iter, f, f_old, opt_tol, prog_tol, f_min)
        if flag is True: break
        # update and project x
        if i == 1:
            np.add(x, -g, x_new) # should update content of x_new
        else:
            t = delta_x.T.dot(delta_g) / delta_g.T.dot(delta_g)
            np.add(x, -t*g, x_new)
        proj(x_new)
        f_new = obj(x_new, g_new) # should update content of g_new
        # do line search between x and x_new, should update x_new, g_new
        f_new = line_search(x, f, g, x_new, f_new, g_new, i)
        # take step
        f_old = f
        f = f_new
        np.add(x_new, -x, delta_x)
        np.add(g_new, -g, delta_g)
        np.copyto(x, x_new)
        np.copyto(g, g_new)
        i += 1
        progress.append([time.time() - start_time, f])
    return {'f': f, 'x': x, 'stop': stop, 'iterations': i, 
            'progress': progress}



def solve_LBFGS(obj, proj, line_search, x_init, f_min=None, opt_tol=1e-6, 
          max_iter=1000, prog_tol=1e-9, corrections=50):
    """Projected batch gradient descent with Barzilei-Bornwein step 
    obj: f,g = obj(x) with f the objective value and g the gradient at x
    proj: w = proj(x)
    line_search: customized line search for the obj function
    x_init: initial point
    f_min: optimal objective value
    opt_tol: stop when f-f_min < opt_tol
    max_iter: stop when iter == max_iter 
    prog_tol: stop when progression in f_old-f < prog_tol
    """
    q_delta_g = deque()
    q_delta_x = deque()
    q_rho = deque()
    # d.append('j') append new entry to the right side
    # d.popleft() return and remove the leftmost item
    t_proj = 0.
    t_obj = 0.
    t_line = 0.
    # allocate memories and initialize
    n = x_init.shape[0]
    x = x_init
    g = np.zeros(n)
    d = np.zeros(n) # search direction computed by LBFGS step
    alpha = np.zeros(corrections) # used in LBFGS
    delta_x = np.zeros(n)
    delta_g = np.zeros(n)
    g_new = np.zeros(n)
    x_new = np.zeros(n)
    f_old = np.inf
    i = 1
    f = obj(x, g) # should update content of g
    progress = []
    start_time = time.time()
    while True:
        #print 'f =', f
        flag, stop = stopping(i, max_iter, f, f_old, opt_tol, prog_tol, f_min)
        if flag is True: break
        # update and project x
        if i == 1:
            np.add(x, -g, x_new) # should update content of x_new
        else:
            # append delta_g, delta_x, rho to queues
            q_delta_g.append(delta_g)
            q_delta_x.append(delta_x)
            q_rho.append(1 / delta_g.T.dot(delta_x))
            if i > corrections+1:
                # only keep the last updates
                q_delta_g.popleft()
                q_delta_x.popleft()
                q_rho.popleft()
            # d = -(delta_x.T.dot(delta_g) / delta_g.T.dot(delta_g)) * g
            LBFGS_helper(q_delta_g, q_delta_x, q_rho, g, d, alpha)
            np.add(x, d, x_new)
        proj(x_new)
        f_new = obj(x_new, g_new) # should update content of g_new
        # do line search between x and x_new, should update x_new, g_new
        # CONSIDER HAVING LINE SEARCH FOR ALL ITERATIONS
        f_new = line_search(x, f, g, x_new, f_new, g_new, i)
        # take step
        f_old = f
        f = f_new
        np.add(x_new, -x, delta_x)
        np.add(g_new, -g, delta_g)
        np.copyto(x, x_new)
        np.copyto(g, g_new)
        i += 1
        progress.append([time.time() - start_time, f])
    return {'f': f, 'x': x, 'stop': stop, 'iterations': i, 
            'progress': progress}


def LBFGS_helper(q_delta_g, q_delta_x, q_rho, g, d, alpha):
    m = len(q_delta_g)
    np.copyto(d, g)
    for j in range(2,m+1):
        alpha[-j] = q_rho[-j] * q_delta_x[-j].T.dot(d)
        d -= alpha[-j] * q_delta_g[-j]

    t = q_delta_x[-1].T.dot(q_delta_g[-1]) / q_delta_g[-1].T.dot(q_delta_g[-1])
    #print 't = ', t
    d *= t

    for j in range(m-1):
        beta = q_rho[j] * q_delta_g[j].T.dot(d)
        d += q_delta_x[j] * (alpha[-m+j] - beta)
    #print '||d|| =', np.linalg.norm(d)
    #print '||d.T * g|| =', abs(d.T.dot(g))
    d *= -1.0


def solve_MD(obj, block_starts, step_size, x_init, line_search=None, f_min=None, opt_tol=1e-6, 
          max_iter=1000, prog_tol=1e-9):
    """mirror descent algorithm
    """
    n = x_init.shape[0]
    block_ends = np.append(block_starts[1:], [n])
    def normalize(x):
        normalization(x, block_starts, block_ends)
    x = x_init
    g = np.zeros(n)
    g_new = np.zeros(n)
    x_new = np.zeros(n)
    f_old = np.inf
    i = 1
    f = obj(x, g) # should update content of g
    progress = []
    start_time = time.time()
    while True:
        flag, stop = stopping(i, max_iter, f, f_old, opt_tol, prog_tol, f_min)
        if flag is True: break
        # update x
        t = step_size(i)
        np.copyto(x_new, x * np.exp(-t*g))
        normalize(x_new)
        f_new = obj(x_new, g_new)
        # take step
        f_old = f
        f = f_new
        np.copyto(x, x_new)
        np.copyto(g, g_new)
        i += 1
        progress.append([time.time() - start_time, f])
    return {'f': f, 'x': x, 'stop': stop, 'iterations': i,
            'progress': progress}

