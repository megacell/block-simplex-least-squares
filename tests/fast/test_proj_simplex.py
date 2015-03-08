import unittest
import sys

sys.path.append('../../')
from python.proj_simplex.proj_simplex_c import proj_simplex_c, proj_multi_simplex_c
from python.proj_simplex.proj_simplex import proj_simplex, proj_multi_simplex

import numpy as np

__author__ = 'jeromethai'


class TestProjSimplex(unittest.TestCase):

    def setUp(self):
        # The setup code is run before each test
        seed = 237423433
        np.random.seed(seed)


    def run_tests_single_block_with(self, proj):
        z = np.array([5.352, 3.23, 32.78, -1.234, 1.7, 104., 53.])
        array1 = [5.352, 3.23, 1., 0., 1.7, 104., 53.]
        array2 = [0., 0., 0., 0., 0, 1., 0.]
        for truth, start, end in zip([array1, array2, z], [2,0,4], [4,7,4]):
            y = np.copy(z)
            proj(y, start, end)
            self.compare_arrays(y, truth)
        y = np.random.rand(7)
        proj(y, 0, 7)
        truth = np.array([0., .05006376, .54108944, 0., .38841272, 0., .02043408])
        self.assertTrue(np.linalg.norm(y - truth) < 1e-6)
        # check out of bound indices
        for start, end in [(2,8), (-1,7), (-1,4)]:
            try:
                proj(y, start, end)
                self.assertTrue(False)
            except AssertionError:
                self.assertTrue(True)

    def run_tests_multiple_block_with(self, proj):
        z = np.array([5.352, 3.23, 32.78, -1.234, 1.7, 104., 53.])
        y = np.copy(z)
        b1, array1 = np.array([0,2,4]), [1., 0., 1., 0., 0., 1., 0.]
        b2, array2 = np.array([0]), [0., 0., 0., 0., 0., 1., 0.]
        b3, array3 = np.array([0,3]), [0., 0., 1., 0., 0., 1., 0.]
        for truth, blocks in zip([array1, array2, array3], [b1, b2, b3]):
            y = np.copy(z)
            proj(y, blocks)
            self.compare_arrays(y, truth)
        for b in [np.array([-1,2,4]), np.array([1,3,7]), np.array([0,4,2])]:
            try:
                proj(y, b)
                self.assertTrue(False)
            except AssertionError:
                self.assertTrue(True)


    def test_proj_simplex_c(self):
        self.run_tests_single_block_with(proj_simplex_c)


    def test_proj_simplex(self):
        self.run_tests_single_block_with(proj_simplex)


    def test_proj_multi_simplex_c(self):
        self.run_tests_multiple_block_with(proj_multi_simplex_c)


    def test_proj_multi_simplex(self):
        self.run_tests_multiple_block_with(proj_multi_simplex)


    def compare_arrays(self, array1, array2):
        for a,b in zip(array1,array2): self.assertTrue(a == b)



if __name__ == '__main__':
    unittest.main()