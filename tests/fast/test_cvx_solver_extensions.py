import unittest
import sys

sys.path.append('../../')
from python.c_extensions.c_extensions import quad_obj
from cvxopt import matrix

import numpy as np

__author__ = 'jeromethai'

class testCvxSolverExtensions(unittest.TestCase):

    def setUp(self):
        seed = 237423433
        np.random.seed(seed)


    def test_quad_obj(self):
    	n = 7
        x = 2*np.random.rand(n) - 1
        Q = 2*np.random.rand(n,n) - 1
        c = 2*np.random.rand(n) - 1
        g = np.zeros(n)
        f, g = quad_obj(x, Q, c, g)
        x, Q, c = matrix(x), matrix(Q), matrix(c)
        f2 = (.5 * x.T * Q * x + c.T * x)[0]
        g2 = Q * x + c
        assert abs(f2-f) < 1e-6
        for i in range(n): assert abs(g[i]-g2[i]) < 1e-6



if __name__ == '__main__':
    unittest.main()