import numpy as np
import time


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
    #t_stop = 0.
    #t_add = 0.
    t_proj = 0.
    t_obj = 0.
    t_line = 0.
    #t_copy = 0.
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
        # stopping criteria
        start_time = time.time()
        if i == max_iter:
            stop = 'stop with max_iter'; break
        if f_min is not None and f-f_min < opt_tol:
            stop = 'stop with f-f_min < opt_tol'; break
        if f_old-f < prog_tol:
            stop = 'stop with f_old-f < prog_tol'; break
        #t_stop += time.time() - start_time 
        # update and project x
        #start_time = time.time()
        np.add(x, -g, x_new) # should update content of x_new
        #x_new = x-g
        #t_add += time.time() - start_time
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
        #start_time = time.time()
        f_old = f
        f = f_new
        np.copyto(x, x_new)
        np.copyto(g, g_new)
        #x = x_new
        #g = g_new
        #t_copy += time.time() - start_time
        i += 1
    return {'f': f, 'x': x, 'stop': stop, 'iterations': i, 
            't_proj': t_proj, 't_obj': t_obj, 't_line': t_line}



