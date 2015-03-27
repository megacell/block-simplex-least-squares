import unittest
import time

__author__ = 'jeromethai'

import sys
sys.path.append('../../')
from python.algorithm_utils import proj_PAV
from python.c_extensions.c_extensions import (isotonic_regression_c,
                                              isotonic_regression_multi_c,
                                              isotonic_regression_c_2,
                                              isotonic_regression_multi_c_2,
                                              isotonic_regression_c_3,
                                              isotonic_regression_multi_c_3)
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state

class TestStressIsotonicRegression(unittest.TestCase):
    
    # def setUp(self):
    #     # The setup code is run before each test
    #     seed = 237423433
    #     np.random.seed(seed)


    def test_isotonic_regression(self):
        self.setUp()
        times = []
        rs = check_random_state(0)
        for n in [int(1e1), int(1e2), int(1e3), int(1e4)]:
            x = np.arange(n)
            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
            ir = IsotonicRegression()
            start_time = time.time()
            y1 = ir.fit_transform(x, y)
            times.append(time.time() - start_time)
        print 'test isotonic_regression'
        print times


    def test_proj_PAV(self):
        self.setUp()
        times = []
        rs = check_random_state(0)
        for n in [int(1e3), int(1e4)]:
            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
            start_time = time.time()
            proj_PAV(y)
            times.append(time.time() - start_time)
        print 'test proj_PAV'
        print times


    def test_isotonic_regression_c(self):
        self.setUp()
        times = []
        rs = check_random_state(0)
        for n in [int(1e1), int(1e2), int(1e3), int(1e4), int(1e5), int(1e6), int(1e7)]:
            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
            start_time = time.time()
            isotonic_regression_c(y, 0, n, update=0)
            times.append(time.time() - start_time)
        print 'test isotonic_regression_c'
        print times


    def test_isotonic_regression_multi_c(self):
        self.setUp()
        n = int(1e6)
        times = []
        rs = check_random_state(0)
        for num_blocks in [int(1e1), int(1e2), int(1e3), int(1e4)]:
            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
            blocks = np.sort(np.random.choice(n, num_blocks, replace=False))
            start_time = time.time()
            isotonic_regression_multi_c(y, blocks, update=0)
            times.append(time.time() - start_time)
        print 'test isotonic_regression_multi_c'
        print times


    def test_isotonic_regression_c_2(self):
        self.setUp()
        times = []
        rs = check_random_state(0)
        for n in [int(1e1), int(1e2), int(1e3), int(1e4), int(1e5), int(1e6), int(1e7)]:
            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
            start_time = time.time()
            isotonic_regression_c_2(y, 0, n)
            times.append(time.time() - start_time)
        print 'test isotonic_regression_c_2'
        print times


    def test_isotonic_regression_multi_c_2(self):
        self.setUp()
        n = int(1e6)
        times = []
        rs = check_random_state(0)
        for num_blocks in [int(1e1), int(1e2), int(1e3), int(1e4)]:
            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
            blocks = np.sort(np.random.choice(n, num_blocks, replace=False))
            start_time = time.time()
            isotonic_regression_multi_c_2(y, blocks)
            times.append(time.time() - start_time)
        print 'test isotonic_regression_multi_c_2'
        print times


    def test_isotonic_regression_c_3(self):
        self.setUp()
        times = []
        rs = check_random_state(0)
        for n in [int(1e1), int(1e2), int(1e3), int(1e4), int(1e5), int(1e6), int(1e7)]:
            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
            start_time = time.time()
            isotonic_regression_c_3(y, 0, n, update=0)
            times.append(time.time() - start_time)
        print 'test isotonic_regression_c_3'
        print times


    def test_isotonic_regression_multi_c_3(self):
        self.setUp()
        n = int(1e6)
        times = []
        rs = check_random_state(0)
        for num_blocks in [int(1e1), int(1e2), int(1e3), int(1e4)]:
            y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
            blocks = np.sort(np.random.choice(n, num_blocks, replace=False))
            start_time = time.time()
            isotonic_regression_multi_c_3(y, blocks, update=0)
            times.append(time.time() - start_time)
        print 'test isotonic_regression_multi_c_3'
        print times


    def test_isotonic_regression_worst_case(self):
        times1 = []
        times2 = []
        times3 = []
        for n in [int(1e1), int(1e2), int(1e3), int(1e4), int(1e5)]:
            y = np.arange(n).astype(float)
            y[-1] = -1e5
            start_time = time.time()
            isotonic_regression_c_3(np.copy(y), 0, n, update=0)
            times3.append(time.time() - start_time)
            start_time = time.time()
            isotonic_regression_c_2(np.copy(y), 0, n)
            times2.append(time.time() - start_time)
            start_time = time.time()
            isotonic_regression_c(np.copy(y), 0, n, update=0)
            times1.append(time.time() - start_time)
        print 'worst case times'
        print 'iso1', times1
        print 'iso2', times2
        print 'iso3', times3




if __name__ == '__main__':
    unittest.main()


