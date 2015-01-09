import numpy as np
import numpy.linalg as la
import time
import math
# Weak Wolfe line search
# t: step size
# d: search direction
# x: current position
def weak_wolfe_ls(x,d,f,nabla_f,proj=lambda x: x, c1=1e-3,c2=0.9):
    # TODO projected version is much slower but doesn't seem to aid performance?

    i = 1
    alpha, beta = 0, float('inf')
    t = 1
    stop = False
    # bisection method
    # https://www.math.washington.edu/~burke/crs/408/notes/nlp/line.pdf [Page 8]
    proj_x = proj(x)
    nabla_fx = nabla_f(proj_x)
    while not stop:
        # print t, x
        proj_xtd = proj(x + t*d)

        i += 1
        # if i >= 1000:
        #     print np.abs(alpha-beta), t, alpha, beta

        # armijo condition violated
        if f(proj_xtd) >= f(proj_x) + c1 * t * d.dot(nabla_fx):
            # print 'ar', f(proj_xtd),f(proj_x)+c1*t*d.dot(nabla_fx),f(proj_x),alpha,beta,t
            # print nabla_fx
            beta = t
            t = 0.5 * (alpha + beta)
        # curvature condition violated
        elif d.dot(nabla_f(proj_xtd)) < c2 * d.dot(nabla_fx):
            # print 'cc', d.dot(nabla_f(proj_xtd)),c2*d.dot(nabla_fx),alpha,beta,t
            alpha = t
            t = 2 * alpha if beta == float('inf') else 0.5 * (alpha + beta)
        # both conditions pass
        else:
            stop = True

        if np.abs(alpha-beta) <= 1e-14:
            stop = True
        
        if la.norm(t*d) <= 1e-8:
            stop = True

        if np.abs(alpha-beta) == 0:
            import ipdb
            ipdb.set_trace()
        # print np.abs(alpha-beta)
    return t

# Limited memory BFGS
def solve(x0, f, nabla_f, stopping, m=50,record_every=500,proj=None, log=None,
        options=None):
    # FIXME has issues when x == 0 to start

    def search_dir(g_new,y_new,s_new,rho,y,s,m=10):
        q = g_new
        alpha = [0]*m
        for i in xrange(m-1,-1,-1):
            alpha[i] = rho[i] * (s[i].dot(q))
            q = q - alpha[i] * y[i]
        H = y_new.dot(s_new) / (y_new.dot(y_new))
        r = H * q
        for i in xrange(0,m):
            beta = rho[i] * y[i].dot(r)
            r = r + s[i] * (alpha[i] - beta)
        return -r

    # Save initial state
    start = log(0,x0,0)

    # Initializations
    i,stop = 0,False
    x = x0
    n = x.shape[0]
    y, s = [np.zeros((n))]*m, [np.zeros((n))]*m
    g_new = nabla_f(x)
    y_new, s_new = g_new, np.ones((n))

    rho, rho_new = [0]*m, 1 / (y_new.dot(s_new))
    while not stop:
        i+=1
        d = search_dir(g_new,y_new,s_new,rho,y,s,m=m)
        # update history
        y.pop(0)
        y.append(y_new)
        s.pop(0)
        s.append(s_new)
        rho.pop(0)
        rho.append(rho_new)

        # update position
        t = weak_wolfe_ls(x,d,f,nabla_f,proj=proj)
        s_new = t * d
        x_next = x + s_new
        if proj:
            x_next = proj(x_next)
        g = g_new
        g_new = nabla_f(x_next)
        y_new = g_new - g 
        if y_new.dot(s_new) == 0:
            print "iter=%d, f=%8.5e" % (i,f(x_next))
            print "Exiting... no change in gradient"
            break
        rho_new = 1 / (y_new.dot(s_new))

        x = x_next
        fx = f(x)
        if (math.isnan(fx)):
            raise ArithmeticError("objective function evaluates to NaN")
        stop = stopping(g_new,fx,i,t,d=d,options=options)

        # Save intermediate state
        if i % record_every == 0:
            start = log(i,x,time.time()-start)

    # Save final state
    log(i,x,time.time()-start)
    return x
