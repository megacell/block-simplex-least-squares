'''
Created on 6 mar. 2015

@author: jerome thai
'''

from c_extensions import isotonic_regression_c, isotonic_regression_multi_c, proj_simplex_c, proj_multi_simplex_c, quad_obj
import numpy as np

y = np.array([4.,5.,1.,6.,8.,7.])
w = np.ones(6)
isotonic_regression_c(y, w, 0, 6)
truth = np.array([3.33333333, 3.33333333, 3.33333333, 6., 7.5, 7.5])
assert np.linalg.norm(y-truth) < 1e-6

y = np.array([4.,5.,1.,6.,8.,7.])
blocks = np.array([0, 2, 4])
isotonic_regression_multi_c(y, w, blocks)
truth = np.array([4., 5., 1., 6., 7.5, 7.5])
assert np.linalg.norm(y-truth) < 1e-6

y = np.array([5.352, 3.23, 32.78, -1.234, 1.7, 104., 53.])
# projects the subvector y at [2:4] into the l1-ball 
proj_simplex_c(y, 2, 4)
for i,e in enumerate([5.352, 3.23, 1., 0., 1.7, 104., 53.]):
    assert y[i] == e


blocks = np.array([0,2,4])
y2 = np.array([5.352, 3.23, 32.78, -1.234, 1.7, 104., 53.])
# projects slices of y2 at [0:2], [2:4], [4:] onto the l1-ball 
proj_multi_simplex_c(y2, blocks)
for i,e in enumerate([1., 0., 1., 0., 0., 1., 0.]):
    assert y2[i] == e


x = np.array([4.4, 5.3])
Q = 2 * np.array([[2, .5], [.5, 1]])
c = np.array([1.0, 1.0])
g = np.zeros(2)
f, g = quad_obj(x, Q, c, g)
assert abs(f - 99.83) < 1e-6 # f = .5*x'*Q*x + c'*x
assert abs(g[0] - 23.9) + abs(g[1] - 16.0) < 1e-6 # g = Q*x + c


print 'Yay! c_extensions work fine!'