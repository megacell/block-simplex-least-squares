
'''
Created on 5 nov. 2014

@author: jerome thai
'''

from proj_l1ball_c import proj_l1ball_c, proj_multi_l1ball_c
import timeit
import numpy as np


y = np.array([5.352, 3.23, 32.78, -1.234, 1.7, 104., 53.])
proj_l1ball_c(y, 2, 4)
for i,e in enumerate([5.352, 3.23, 1., 0., 1.7, 104., 53.]):
    assert y[i] == e


blocks = np.array([0.,2.,4.])
y2 = np.array([5.352, 3.23, 32.78, -1.234, 1.7, 104., 53.])
proj_multi_l1ball_c(y2, blocks)
for i,e in enumerate([1., 0., 1., 0., 0., 1., 0.]):
    assert y2[i] == e