import unittest
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
import sys

sys.path.append('../../')
from python.c_extensions.python_implementation import proj_PAV
from python.c_extensions.c_extensions import isotonic_regression_c, isotonic_regression_multi_c

import numpy as np

__author__ = 'jeromethai'

class TestIsotonicRegression(unittest.TestCase):


    def sklearn_isotonic_regression_multi(self, y, blocks):
        ir = IsotonicRegression()
        n = len(y)
        x = np.arange(n)
        z = np.zeros(n)
        for start, end in zip(blocks, np.append(blocks[1:], [n])):
            z[start:] = y[start:]
            z[start:end] = ir.fit_transform(x[start:end], y[start:end])
        return z


    def test_isotonic_regression_c(self):
        n = 10
        x = np.arange(n)
        rs = check_random_state(0)
        for i in range(10):
            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
            ir = IsotonicRegression()
            truth = ir.fit_transform(x, y)
            self.assertTrue(np.linalg.norm(isotonic_regression_c(y,0,n) - truth) < 1e-8)
    

    def test_proj_PAV(self):
        n = 10
        x = np.arange(n)
        rs = check_random_state(0)
        for i in range(10):
            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
            ir = IsotonicRegression()
            truth = ir.fit_transform(x, y)
            self.assertTrue(np.linalg.norm(proj_PAV(y) - truth) < 1e-8)


    def test_isotonic_regression_multi_c(self):
        n = 10
        rs = check_random_state(0)
        for i in range(10):
            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
            blocks = np.sort(np.random.choice(n, 3, replace=False))
            truth = self.sklearn_isotonic_regression_multi(y, blocks)
            isotonic_regression_multi_c(y, blocks)
            #print np.linalg.norm(y-truth)
            #assert np.linalg.norm(y-truth) < 1e-8



if __name__ == '__main__':
    unittest.main()