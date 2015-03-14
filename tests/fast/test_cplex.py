"""
Example:
Concider the problem
0.5 * (x1^2 + 2x2^2 + 3x3^2) + 15x1 + 8x2 + 80x3 -> min        (1)
subjected to
x1 + 2x2 + 3x3 <= 150            (2)
8x1 +  15x2 +  80x3 <= 800    (3)
x2 - x3 = 25.5                           (4)
x1 <= 15                                  (5)
"""

import unittest
import pdb

import numpy as np
from numpy import diag, matrix, inf
from openopt import QP

class TestCplex(unittest.TestCase):
    def test_cplex(self):
        p = QP(diag([1, 2, 3]),
               [15, 8, 80],
               A = matrix('1 2 3; 8 15 80'),
               b = [150, 800],
               Aeq = [0, 1, -1],
               beq = 25.5,
               ub = [15,inf,inf])
        r = p._solve('cplex', iprint = 0)
        f_opt, x_opt = r.ff, r.xf
        np.testing.assert_almost_equal(f_opt,  -1190.35)
        np.testing.assert_almost_equal(x_opt, [-15. ,  -2.3, -27.8 ])

if __name__ == '__main__':
    unittest.main()
