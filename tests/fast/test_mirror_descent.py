import unittest

import numpy as np
import sys
sys.path.append('../../')
import python.mirror_descent as md
from python.algorithm_utils import quad_obj_np, 


class TestMirrorDescent(unittest.TestCase):

    def setUp(self):
        pass

    def test_least_squares_one_block(self):
        one_block_soln = md.least_squares(
                np.array([[1, 0], [0, 1]]),
                np.array([1, 1.5]),
                np.array([2]))
        np.testing.assert_almost_equal(np.array([0.25, 0.75]), one_block_soln)

    def test_least_squares_two_blocks(self):
        two_block_simple_soln = md.least_squares(
                np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]),
                np.array([1, 1.5, 1, 1.5]),
                np.array([2, 2]))
        np.testing.assert_almost_equal(np.array([0.25, 0.75, 0.25, 0.75]),
                two_block_simple_soln)

    def test_least_squares_complex(self):
        A = np.array([[1, 1, 3, 6],
                      [9, 1, 8, 2],
                      [1, 7, 1, 7],
                      [0, 0, 0, 8]])
        x = np.squeeze(np.array([0.2, 0.8, 0.78, 0.22]))
        b = A.dot(x)
        two_block_soln = md.least_squares(A, b, np.array([2, 2]))
        np.testing.assert_almost_equal(x,
                two_block_soln)

    def test_least_squares_uneven_blocks(self):
        A = np.array([[1, 1, 3, 6, 4],
                      [9, 1, 8, 2, 2],
                      [1, 7, 1, 7, 1],
                      [0, 0, 0, 8, 5],
                      [0, 2, 0, 8, 5]])
        x = np.squeeze(np.array([0.2, 0.8, 0.15, 0.40, 0.45]))
        b = A.dot(x)
        two_block_soln = md.least_squares(A, b, np.array([2, 3]))
        np.testing.assert_almost_equal(x,
                two_block_soln)


    def test_mirror(self):
        Q = 2 * np.array([[2, .5], [.5, 1]])
        c = np.array([1.0, 1.0])
        x_true = np.array([.25, .75])

    def obj(x, g):
        return quad_obj_np(x, Q, c, g)


     x0 = np.array([.5, .5])
     f_min  = 1.875

     sol = md.solve(obj, block_starts, x0)
     print sol['x']



if __name__=='__main__':
    unittest.main()
