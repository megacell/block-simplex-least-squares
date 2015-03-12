import numpy as np
import time


def stopping(i, max_iter, f, f_old, opt_tol, prog_tol, f_min=None):
    flag = False
    stop = 'continue'
    if i == max_iter:
        stop = 'max_iter';
        flag = True
    if f_min is not None and f-f_min < opt_tol:
        stop = 'f-f_min = {} < opt_tol'.format(f-f_min)
        flag = True
    if abs(f_old-f) < prog_tol:
        stop = '|f_old-f| = {} < prog_tol'.format(abs(f_old-f))
        flag = True
    return flag, stop



def solve(obj, proj, line_search, x0, f_min=None, opt_tol=1e-6, 
          max_iter=1000, prog_tol=1e-9):
    """Projected batch gradient descent with line search
    obj: f,g = obj(x) with f the objective value and g the gradient at x
    proj: w = proj(x)
    line_search: customized line search for the obj function
    x0: initial point
    f_min: optimal objective value
    opt_tol: stop when f-f_min < opt_tol
    max_iter: stop when iter == max_iter 
    prog_tol: stop when progression in f_old-f < prog_tol
    """
    t_proj = 0.
    t_obj = 0.
    t_line = 0.
    # allocate memories and initialize
    n = x0.shape[0]
    x = x0
    g = np.zeros(n)
    g_new = np.zeros(n)
    x_new = np.zeros(n)
    f_old = np.inf
    i = 1
    f = obj(x, g) # should update content of g
    while True:
        flag, stop = stopping(i, max_iter, f, f_old, opt_tol, prog_tol, f_min)
        if flag is True: break
        # update and project x
        np.add(x, -g, x_new) # should update content of x_new
        start_time = time.time()
        proj(x_new)
        t_proj += time.time() - start_time
        start_time = time.time()
        f_new = obj(x_new, g_new) # should update content of g_new
        t_obj += time.time() - start_time
        # do line search between x and x_new, should update x_new, g_new
        start_time = time.time()
        f_new = line_search(x, f, g, x_new, f_new, g_new, i)
        t_line += time.time() - start_time
        # take step
        f_old = f
        f = f_new
        np.copyto(x, x_new)
        np.copyto(g, g_new)
        i += 1
    return {'f': f, 'x': x, 'stop': stop, 'iterations': i, 
            't_proj': t_proj, 't_obj': t_obj, 't_line': t_line}


def solve_BB(obj, proj, line_search, x0, f_min=None, opt_tol=1e-6, 
          max_iter=1000, prog_tol=1e-9, Q=None):
    """Projected batch gradient descent with Barzilei-Bornwein step 
    obj: f,g = obj(x) with f the objective value and g the gradient at x
    proj: w = proj(x)
    line_search: customized line search for the obj function
    x0: initial point
    f_min: optimal objective value
    opt_tol: stop when f-f_min < opt_tol
    max_iter: stop when iter == max_iter 
    prog_tol: stop when progression in f_old-f < prog_tol
    """
    t_proj = 0.
    t_obj = 0.
    t_line = 0.
    # allocate memories and initialize
    n = x0.shape[0]
    x = x0
    g = np.zeros(n)
    delta_x = np.zeros(n)
    delta_g = np.zeros(n)
    g_new = np.zeros(n)
    x_new = np.zeros(n)
    f_old = np.inf
    i = 1
    f = obj(x, g) # should update content of g
    while True:
        flag, stop = stopping(i, max_iter, f, f_old, opt_tol, prog_tol, f_min)
        if flag is True: break
        # update and project x
        if i == 1:
            np.add(x, -g, x_new) # should update content of x_new
        else:
            t = delta_x.T.dot(delta_g) / delta_g.T.dot(delta_g)
            np.add(x, -t*g, x_new)
        start_time = time.time()
        proj(x_new)
        t_proj += time.time() - start_time
        start_time = time.time()
        f_new = obj(x_new, g_new) # should update content of g_new
        t_obj += time.time() - start_time
        # do line search between x and x_new, should update x_new, g_new
        start_time = time.time()
        if i == 1: f_new = line_search(x, f, g, x_new, f_new, g_new, i)
        t_line += time.time() - start_time
        # take step
        f_old = f
        f = f_new
        np.add(x_new, -x, delta_x)
        np.add(g_new, -g, delta_g)
        np.copyto(x, x_new)
        np.copyto(g, g_new)
        i += 1
    return {'f': f, 'x': x, 'stop': stop, 'iterations': i, 
            't_proj': t_proj, 't_obj': t_obj, 't_line': t_line}

