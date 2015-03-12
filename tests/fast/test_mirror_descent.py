import unittest

import numpy as np
import sys
sys.path.append('../../')
import python.mirror_descent as md
from python.algorithm_utils import quad_obj_np
from python.bsls_utils import almost_equal
import cvxopt as copt


__author__ = 'jeromethai', 'yuanchenyang'

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

        block_starts = np.array([0])
        x0 = np.array([.5, .5])
        f_min  = 1.875

        sol = md.solve(obj, block_starts, x0)
        assert almost_equal(sol['x'], x_true, 1e-5)
        assert sol['stop'][-10:] == '< prog_tol'

        sol = md.solve(obj, block_starts, x0, f_min=f_min)
        assert almost_equal(sol['x'], x_true, 1e-5)
        assert sol['stop'][-10:] == ' < opt_tol'


    def test_batch_2(self):
        # test on constrainted least squares problem
        # min ||Ax-b||^2 = x'A'Ax - 2 b'Ax
        # s.t. ||x||_1 = 1, x>=0
        for i in range(5):
            m = 10 # number of measurements
            n = 7 # dimension of features
            block_starts = np.array([0])
            A = np.random.randn(m, n)
            x_true = abs(np.random.randn(n,1))
            x_true = x_true / np.linalg.norm(x_true, 1)
            b = A.dot(x_true)
            Q = A.T.dot(A)
            c = -A.T.dot(b)
            c = c.flatten()
            x0 = np.ones(n) / n

            def obj(x, g):
                return quad_obj_np(x, Q, c, g)
                
            x_true = x_true.flatten()
            sol = md.solve(obj, block_starts, x0, prog_tol=1e-14)
            print 'precision:', np.linalg.norm(sol['x']-x_true)
            print 'MD does not converge. Choose better step size or projects?'
            #assert almost_equal(sol['x'], x_true, 1e-4)



if __name__=='__main__':
    unittest.main()
