'''
Created on 6 mar. 2015

@author: jerome thai
'''

from isotonic_regression_c import isotonic_regression_c
import numpy as np

y = np.array([4.,5.,1.,6.,8.,7.])
w = np.array([1.,1.,1.,1.,1.,1.])
solution = np.copy(y)
isotonic_regression_c(y, w, solution)
truth = np.array([3.33333333, 3.33333333, 3.33333333, 6., 7.5, 7.5])
assert np.linalg.norm(solution-truth) < 1e-6
print 'Congrats! isotonic_regression_c is properly set up!'