import unittest

__author__ = 'jeromethai'

import sys
sys.path.append('../../python/proj_l1ball/')
from proj_l1ball_c import proj_l1ball_c, proj_multi_l1ball_c
from proj_l1ball import proj_l1ball
import numpy as np


class TestProjL1Ball(unittest.TestCase):


    def run_tests_with(self, proj):
        y = np.array([5.352, 3.23, 32.78, -1.234, 1.7, 104., 53.])
        proj(y, 2, 4)
        for i,e in enumerate([5.352, 3.23, 1., 0., 1.7, 104., 53.]):
            self.assertTrue(y[i] == e)
        
        y = np.array([5.352, 3.23, 32.78, -1.234, 1.7, 104., 53.])
        proj(y, 0, 7)
        for i,e in enumerate([0., 0., 0., 0., 0, 1., 0.]):
            self.assertTrue(y[i] == e)

        y = np.array([5.352, 3.23, 32.78, -1.234, 1.7, 104., 53.])
        proj(y, 4, 4)
        for i,e in enumerate([5.352, 3.23, 32.78, -1.234, 1.7, 104., 53.]):
            self.assertTrue(y[i] == e)

        try:
            proj(y, 2, 8)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)
        try:
            proj(y, -1, 7)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)
        try:
            proj(y, -1, 4)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)


    def test_proj_l1ball_c(self):
        self.run_tests_with(proj_l1ball_c)


    def test_proj_l1ball(self):
    	self.run_tests_with(proj_l1ball)


    def test_proj_multi_l1ball_c(self):
    	#TODO
        pass


if __name__ == '__main__':
    unittest.main()