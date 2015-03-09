import ipdb
import unittest
import random

import numpy as np
import numpy.linalg as la

import python.bsls_utils as util
from python.bsls_matrices import BSLSMatrices

__author__ = 'cathywu'


class TestBSLSMatrices(unittest.TestCase):
    def setUp(self):
        seed = 237423433
        random.seed(seed)
        np.random.seed(seed)
        self.eps = 1e-10
        self.config = {
            'full': True, 'L': True, 'OD': True, 'CP': True,
            'LP': True, 'eq': 'CP', 'init': False,
            }

    def test_simple_simplex_form(self):
        data = util.generate_data()
        bm = BSLSMatrices(data=data, **self.config)

        # Check feasibility after each step
        bm.consolidate(eq=self.config['eq'])
        self.assertTrue(la.norm(bm.C.dot(bm.x_true) - bm.d) < self.eps)
        self.assertTrue(la.norm(bm.AA.dot(bm.x_true) - bm.bb) < self.eps)
        self.assertTrue(np.all(bm.x_true >= 0))

        ones = np.ones(bm.d.shape)

        bm.standard_simplex_form()
        self.assertTrue(la.norm(bm.C.dot(bm.x_split) - ones) < self.eps)
        self.assertTrue(la.norm(bm.AA.dot(bm.x_split) - bm.bb) < self.eps)
        self.assertTrue(np.all(bm.x_split >= 0))

        bm.cleanup()
        self.assertTrue(la.norm(bm.C.dot(bm.x_split) - ones) < self.eps)
        self.assertTrue(la.norm(bm.AA.dot(bm.x_split) - bm.bb) < self.eps)
        self.assertTrue(np.all(bm.x_split >= 0))

        bm.blockify()
        self.assertTrue(la.norm(bm.C.dot(bm.x_split) - ones) < self.eps)
        self.assertTrue(la.norm(bm.AA.dot(bm.x_split) - bm.bb) < self.eps)
        self.assertTrue(np.all(bm.C.nonzero()[1] == np.array(range(bm.x_true.size))))
        self.assertTrue(np.all(bm.x_split >= 0))

        # Check blockwise sums of x and x_split
        cum_blocks = np.concatenate(([0], np.cumsum(bm.block_sizes)))
        blocks_start = cum_blocks
        blocks_end = cum_blocks[1:]
        self.assertTrue(abs(sum(bm.x_split) - data['U'].shape[0]) < self.eps)
        for i, (s, e) in enumerate(zip(blocks_start, blocks_end)):
            self.assertTrue(abs(sum(bm.x_true[s:e])-bm.d[i]) < self.eps)
            self.assertTrue(abs(sum(bm.x_split[s:e])-1) < self.eps)

    def test_reconstruct(self):
        data = util.generate_data()

        # zero out the first row of A
        n = data['x_true'].size
        data['A'][0,:] = np.zeros(n)

        # zero out the first block (f)
        data['f'][0] = 0
        data['x_true'] = (data['U'].T.dot(data['f']) > 0) * data['x_true']
        data['b'] = data['A'].dot(data['x_true'])

        # build matrices and do the usual tests (on the modified input)
        bm = BSLSMatrices(data=data, **self.config)
        bm.degree_reduced_form()
        ones = np.ones(bm.d.shape)
        self.assertTrue(la.norm(bm.C.dot(bm.x_split) - ones) < self.eps)
        self.assertTrue(la.norm(bm.AA.dot(bm.x_split) - bm.bb) < self.eps)
        self.assertTrue(np.all(bm.x_split >= 0))

        # then test the reconstruction
        AA, bb, _, _, x_split, nz, scaling, rsort_index, x0 = bm.get_LS()
        self.assertTrue(AA.shape[0] == data['A'].shape[0]-1)
        x_true = bm.reconstruct(x_split, rsort_index=rsort_index,
                                scaling=scaling, nz=nz, n=data['x_true'].size)
        self.assertTrue(la.norm(data['x_true'] - x_true) < self.eps)

    def test_get_LS(self):
        data = util.generate_data()
        bm = BSLSMatrices(data=data, **self.config)
        bm.degree_reduced_form()
        AA, bb, N, block_sizes, x_split, nz, scaling, rsort_index, x0 = bm.get_LS()
        self.assertTrue(la.norm(AA.dot(x_split) - bb) < self.eps)

        cum_blocks = np.concatenate(([0], np.cumsum(block_sizes)))
        blocks_start = cum_blocks
        blocks_end = cum_blocks[1:]
        self.assertTrue(abs(sum(x_split) - data['U'].shape[0]) < self.eps)
        for (s,e) in zip(blocks_start, blocks_end):
            self.assertTrue(abs(sum(x_split[s:e])-1) < self.eps)

        x_true = bm.reconstruct(x_split, rsort_index=rsort_index,
                                scaling=scaling, nz=nz, n=data['x_true'].size)
        self.assertTrue(la.norm(data['x_true'] - x_true) < self.eps)

        # TODO test N

if __name__ == '__main__':
    unittest.main()
