
'''
Created on 5 nov. 2014

@author: jerome thai
'''

from proj_simplex_c import proj_simplex_c, proj_multi_simplex_c
import numpy as np


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