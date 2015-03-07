import unittest
import sys
sys.path.append('../../python/proj_simplex/')
from proj_simplex_c import proj_simplex_c, proj_multi_simplex_c
from proj_simplex import proj_simplex
import numpy as np


__author__ = 'jeromethai'


class TestProjSimplex(unittest.TestCase):


    def run_tests_single_block_with(self, proj):
        z = np.array([5.352, 3.23, 32.78, -1.234, 1.7, 104., 53.])
        array1 = [5.352, 3.23, 1., 0., 1.7, 104., 53.]
        array2 = [0., 0., 0., 0., 0, 1., 0.]
        for truth, start, end in zip([array1, array2, z], [2,0,4], [4,7,4]):
            y = np.copy(z)
            proj(y, start, end)
            self.compare_arrays(y, truth)
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
        blocks = np.array([0,3,4])
        print 'multi-proj'
        proj(y, blocks)
        print y


    def test_proj_simplex_c(self):
        self.run_tests_single_block_with(proj_simplex_c)


    def test_proj_simplex(self):
    	self.run_tests_single_block_with(proj_simplex)


    def test_proj_multi_simplex_c(self):
        self.run_tests_multiple_block_with(proj_multi_simplex_c)
        


    def compare_arrays(self, array1, array2):
        for a,b in zip(array1,array2): self.assertTrue(a == b)



if __name__ == '__main__':
    unittest.main()