import unittest
import time

__author__ = 'jeromethai'

import sys
sys.path.append('../../')

from python.c_extensions.c_extensions import proj_simplex_c, proj_multi_simplex_c
from python.algorithm_utils import proj_simplex, proj_multi_simplex

import numpy as np


class TestStressProjSimplex(unittest.TestCase):


    def setUp(self):
        # The setup code is run before each test
        seed = 237423433
        np.random.seed(seed)


    def run_tests_single_block_with(self, proj):
        self.setUp()
        times = []
        for n in [int(1e3), int(1e4), int(1e5), int(1e6)]:
            y = np.random.rand(n)
            start_time = time.time()
            proj(y, 0, n)
            times.append(time.time() - start_time)
        return times


    def run_tests_multi_blocks_with(self, proj):
        self.setUp()
        times = []
        for num_blocks in [int(1e1), int(1e2), int(1e3), int(1e4)]:
            y = np.random.rand(1e6)
            blocks = np.sort(np.random.choice(int(1e6), num_blocks, replace=False))
            start_time = time.time()
            proj(y, blocks)
            times.append(time.time() - start_time)
        return times


    def test_proj_simplex_c(self):
        print "test proj_simplex_c:"
        print self.run_tests_single_block_with(proj_simplex_c)


    def test_proj_simplex(self):
        print "test proj_simplex:"
        print self.run_tests_single_block_with(proj_simplex)


    def test_proj_multi_simplex_c(self):
        print "test proj_multi_simplex_c:"
        print self.run_tests_multi_blocks_with(proj_multi_simplex_c)


    def test_proj_multi_simplex(self):
        print "test proj_multi_simplex:"
        print self.run_tests_multi_blocks_with(proj_multi_simplex)


if __name__ == '__main__':
    unittest.main()