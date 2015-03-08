'''
Created on 6 mar. 2015

@author: jerome thai
'''

from c_extensions import isotonic_regression_c, isotonic_regression_multi_c, proj_simplex_c, proj_multi_simplex_c
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

print 'Yay! proj_simplex_c and proj_multi_simplex_c work fine!'