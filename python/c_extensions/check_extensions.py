'''
Created on 6 mar. 2015

@author: jerome thai
'''

from c_extensions import isotonic_regression_c, isotonic_regression_multi_c, proj_simplex_c, proj_multi_simplex_c, quad_obj, line_search_quad_obj
import numpy as np

print 'test isotonic_regression_c'
y = np.array([4.,5.,1.,6.,8.,7.])
w = np.ones(6)
isotonic_regression_c(y, w, 0, 6)
truth = np.array([3.33333333, 3.33333333, 3.33333333, 6., 7.5, 7.5])
assert np.linalg.norm(y-truth) < 1e-6

print 'test isotonic_regression_multi_c'
y = np.array([4.,5.,1.,6.,8.,7.])
blocks = np.array([0, 2, 4])
isotonic_regression_multi_c(y, w, blocks)
truth = np.array([4., 5., 1., 6., 7.5, 7.5])
assert np.linalg.norm(y-truth) < 1e-6

print 'test proj_simplex_c'
y = np.array([5.352, 3.23, 32.78, -1.234, 1.7, 104., 53.])
# projects the subvector y at [2:4] into the l1-ball 
proj_simplex_c(y, 2, 4)
for i,e in enumerate([5.352, 3.23, 1., 0., 1.7, 104., 53.]):
    assert y[i] == e

print 'test proj_multi_simplex_c'
blocks = np.array([0,2,4])
y2 = np.array([5.352, 3.23, 32.78, -1.234, 1.7, 104., 53.])
# projects slices of y2 at [0:2], [2:4], [4:] onto the l1-ball 
proj_multi_simplex_c(y2, blocks)
for i,e in enumerate([1., 0., 1., 0., 0., 1., 0.]):
    assert y2[i] == e

print 'test quad_obj'
x = np.array([4.4, 5.3])
Q = 2 * np.array([[2, .5], [.5, 1]])
c = np.array([1.0, 1.0])
g = np.zeros(2)
f, g = quad_obj(x, Q, c, g)
assert abs(f - 99.83) < 1e-6 # f = .5*x'*Q*x + c'*x
assert abs(g[0] - 23.9) + abs(g[1] - 16.0) < 1e-6 # g = Q*x + c

print 'test line_search_quad_obj'
def test_line_search(x, f, g, x_new, f_new, g_new, x_true, f_true, g_true, t_true, Q, c):
    x_new, f_new, g_new, t = line_search_quad_obj(x, f, g, x_new, f_new, g_new, Q, c)
    assert np.linalg.norm(x_new - x_true) < 1e-8
    assert abs(f_new - f_true) < 1e-8
    assert np.linalg.norm(g_new - g_true) < 1e-8
    assert abs(t - t_true) < 1e-8

# parameters of the quadratic objective
Q = 2 * np.array([[2, .5], [.5, 1]])
c = np.array([1.0, 1.0])

# test 1
# current point
x = np.array([0.5, 0.5])
f = 2.
g = np.array([3.5, 2.5])
# next point
x_new = np.array([0., 1.]) 
f_new = 2.
g_new = np.array([2., 3.])
# true values
x_true = np.array([0.25, 0.75]) 
f_true = 1.875
g_true = np.array([2.75, 2.75])
t_true = 0.5
test_line_search(x, f, g, x_new, f_new, g_new, x_true, f_true, g_true, t_true, Q, c)

# test 2
x_new = np.array([0., 1.]) 
f_new = 2.
g_new = np.array([2., 3.])
test_line_search(x_true, f_true, g_true, x_new, f_new, g_new, x_true, f_true, g_true, 0.0, Q, c)

# test 3
# current point
x = np.array([0.26, 0.74]) 
f = 1.8752
g = np.array([2.78, 2.74])
# next point
x_new = np.array([0., 1.]) 
f_new = 2.
g_new = np.array([2., 3.])
# true values
x_true = np.array([0.2559375, 0.7440625])
f_true = 1.87507050781
g_true = np.array([2.7678125, 2.7440625])
t_true = 0.125
test_line_search(x, f, g, x_new, f_new, g_new, x_true, f_true, g_true, t_true, Q, c)


print 'Yay! c_extensions work fine!'
