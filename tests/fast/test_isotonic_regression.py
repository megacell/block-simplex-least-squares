import unittest
from sklearn.isotonic import IsotonicRegression
import sys
sys.path.append('../../python/isotonic_regression/')
from isotonic_regression import proj_PAV

__author__ = 'jeromethai'

import numpy as np


class TestIsotonicRegression(unittest.TestCase):

    def setUp(self):
        # The setup code is run before each test
        seed = 237423433
        np.random.seed(seed)


    def test_proj_PAV(self):
        n = 6
        x = np.arange(n)
        y = np.array([4,5,1,6,8,7])
        truth = [3.33333333, 3.33333333, 3.33333333, 6., 7.5, 7.5]
        ir = IsotonicRegression()
        self.assertTrue(np.linalg.norm(ir.fit_transform(x, y) - truth) < 1e-6)
        self.assertTrue(np.linalg.norm(proj_PAV(y) - truth) < 1e-6)


if __name__ == '__main__':
    unittest.main()