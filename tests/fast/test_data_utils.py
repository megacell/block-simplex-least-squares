import unittest
import numpy as np
import sys
sys.path.append('../../')
from python.data_utils import (find_first_indices, 
                                permute_column, 
                                U_to_block_sizes,
                                swap,
                                push_left,
                                row_with_most_ones,
                                aggregate)

__author__ = 'jeromethai'

class TestDataUtils(unittest.TestCase):


    def assert_equal(self, A, B):
        m, n = A.shape
        for i in range(m):
            for j in range(n):
                assert A[i,j] == B[i,j]

    def test_find_first_indices(self):
        U = np.array([[0., 0., 1., 1., 0.], [1., 1., 0., 0., 0.], [0., 0., 0., 0., 1.]])
        ind = find_first_indices(U)
        assert ind[0] == (2, 2) and ind[1] == (0,2) and ind[2] == (4,1)


    def test_permute_column(self):
        U = np.array([[0.,0.,0.,1.,1.,0.], [1.,1.,1.,0.,0.,0.], [0.,0.,0.,0.,0.,1.]])
        expected = np.array([[1.,1.,0.,0.,0.,0.], [0.,0.,1.,1.,1.,0.], [0.,0.,0.,0.,0.,1.]])
        P = permute_column(U)
        U2 = U * P
        self.assert_equal(U2, expected)


    def test_U_to_block_sizes(self):
        U = np.array([[1.,1.,1.,0.,0.], [0.,0.,0.,1.,1.]])
        block_sizes = U_to_block_sizes(U)
        assert block_sizes[0] == 3
        assert block_sizes[1] == 2


    def test_swap(self):
        A = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
        x_true = np.array([10, 11, 12])
        swap(A, x_true, 0, 2)
        A_expected = np.array([[3, 2, 1],[6, 5, 4],[9, 8, 7]])
        x_true_expected = np.array([12, 11, 10])
        self.assert_equal(A, A_expected)
        for i in range(3): assert x_true[i] == x_true_expected[i]


    def test_push_left(self):
        A = np.array([[1, 2, 3, 4, 5, 6], [0, 1, 1, 1, 0, 1]])
        x_true = np.array([1, 2, 3, 4, 5, 6])
        x_true_expected = np.array([6, 2, 3, 4, 5, 1])
        A_expected = np.array([[6, 2, 3, 4, 5, 1], [1, 1, 1, 1, 0, 0]])
        assert push_left(A, x_true, 0, 6, 1) == 4
        self.assert_equal(A, A_expected)
        for j in range(6): x_true[j] == x_true_expected[j]


    def test_row_with_most_ones(self):
        A = np.array([[0,0], [1,1], [1,0]])
        assert row_with_most_ones(A, 0, 2) == 2


    def test_aggregate(self):
        A = np.array([[0,0,1,0,0], [0,1,0,1,0], [1,0,1,0,1], [1,1,1,0,0]])
        x_true = np.array([1,2,3,4,5])
        tmp = A.dot(x_true)
        aggregate(A, x_true, np.array([0]))
        tmp2 = A.dot(x_true)
        for i in range(4): assert tmp[i] == tmp2[i] # Do better test



if __name__ == '__main__':
    unittest.main()