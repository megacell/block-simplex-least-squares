import unittest
import time

__author__ = 'jeromethai'

import sys
sys.path.append('../../python/proj_l1ball/')
from proj_l1ball_c import proj_l1ball_c, proj_multi_l1ball_c
from proj_l1ball import proj_l1ball
import numpy as np


class TestStressProjL1Ball(unittest.TestCase):


    def setUp(self):
        # The setup code is run before each test
        seed = 237423433
        np.random.seed(seed)


    def run_tests_with(self, proj):
    	times = []
    	for n in [1e3, 1e4, 1e5, 1e6]:
    	    y = np.random.rand(n)
            start_time = time.time()
            proj(y, 0, n)
            times.append(time.time() - start_time)
        return times


    def test_proj_l1ball_c(self):
        print self.run_tests_with(proj_l1ball_c)


    def test_proj_l1ball(self):
    	print self.run_tests_with(proj_l1ball)


if __name__ == '__main__':
    unittest.main()
