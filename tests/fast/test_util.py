import ipdb
import unittest
import random

import numpy as np
import numpy.linalg as la

import python.bsls_utils as util

__author__ = 'lei, cathywu'


class TestUtil(unittest.TestCase):
    def setUp(self):
        seed = 237423433
        random.seed(seed)
        np.random.seed(seed)
        self.eps = 1e-10
        self.config = {
            'full': True, 'L': True, 'OD': True, 'CP': True,
            'LP': True, 'eq': 'CP', 'init': False,
            }

    def test_test(self):
        self.assertFalse(False)

    def test_particular_x0(self):
        block_sizes = np.array([1,2,3,4])
        x0 = util.particular_x0(block_sizes)
        ans = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        self.assertTrue(ans.size == x0.size)
        self.assertTrue(np.all(ans==x0))

    def test_generate_data(self):
        data = util.generate_data()
        self.assertTrue(abs(sum(data['f']) - sum(data['x_true'])) < self.eps)

        data = util.generate_data(A_sparse=0.9)
        self.assertTrue(abs(sum(data['f']) - sum(data['x_true'])) < self.eps)


if __name__ == '__main__':
    unittest.main()
