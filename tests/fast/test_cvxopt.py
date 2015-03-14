"""
Example:
Concider the problem
0.5 * (x1^2 + 2x2^2 + 3x3^2) + 15x1 + 8x2 + 80x3 -> min        (1)
subjected to
x1 + 2x2 + 3x3 <= 150            (2)
8x1 +  15x2 +  80x3 <= 800    (3)
x2 - x3 = 25.5                           (4)
"""

import unittest
import pdb

import numpy as np
import cvxopt
from cvxopt import matrix
from numpy import diag, inf
from openopt import QP

class TestCvxopt(unittest.TestCase):
    def test_cvxopt(self):
        Q = matrix([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        p = matrix([15.0, 8.0, 80.0])
        G = matrix([[1.0, 2.0, 3.0], [8.0, 15.0, 80.0], [0, 0, 0]])
        h = matrix([150.0, 800.0, 0], (3, 1))
        A = matrix([0.0, 1.0, -1.0], (1,3))
        b = matrix(25.5)

        sol = cvxopt.solvers.qp(Q, p, G, h, A, b)

        p = QP(diag([1, 2, 3]),
               [15, 8, 80],
               A = np.matrix("1 2 3; 8 15 80"),
               b = [150, 800],
               Aeq = [0, 1, -1],
               beq = 25.5)
        r = p._solve('cvxopt_qp', iprint = 0)
        f_opt, x_opt = r.ff, r.xf

        np.testing.assert_almost_equal(f_opt,  sol['primal objective'], decimal=5)
        np.testing.assert_almost_equal(x_opt, np.squeeze(sol['x']), decimal=5)

if __name__ == '__main__':
    unittest.main()
