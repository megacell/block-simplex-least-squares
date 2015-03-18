import unittest
import numpy as np
import sys
sys.path.append('../../')
from python.data_utils import find_first_indices, permute_column

__author__ = 'jeromethai'

class TestDataUtils(unittest.TestCase):

    def test_find_first_indices(self):
        U = np.array([[0., 0., 1., 1., 0.], [1., 1., 0., 0., 0.], [0., 0., 0., 0., 1.]])
        ind = find_first_indices(U)
        assert ind[0] == (2, 2) and ind[1] == (0,2) and ind[2] == (4,1)


    def test_permute_column(self):
        U = np.array([[0.,0.,0.,1.,1.,0.], [1.,1.,1.,0.,0.,0.], [0.,0.,0.,0.,0.,1.]])
        expected = np.array([[1.,1.,0.,0.,0.,0.], [0.,0.,1.,1.,1.,0.], [0.,0.,0.,0.,0.,1.]])
        P = permute_column(U)
        U2 = U * P
        print U2
        for i in range(3):
            for j in range(6):
                assert U2[i,j] == expected[i,j]


if __name__ == '__main__':
    unittest.main()