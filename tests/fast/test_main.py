import unittest
import random
import argparse

import numpy as np
import scipy.linalg as sla
import scipy.io

from python.main import main

__author__ = 'cathywu'

class TestMain(unittest.TestCase):
    def setUp(self):
        # The setup code is run before each test
        fname = 'test_main.mat'
        seed = 237423433
        random.seed(seed)
        np.random.seed(seed)

        args = argparse.Namespace()
        args.noise = 0
        args.file = fname
        args.log = 'WARN'
        args.init = False
        args.eq = 'CP'
        args.method = 'BB'

        self.args = args

    def generate_data(self, fname, n=100, m1=5, m2=10, A_sparse=0.5, alpha=1.0):
        """
        A is m1 x n
        U is m2 x n

        :param fname: file to save to on disk
        :param n: size of x
        :param m1: number of measurements
        :param m2: number of blocks
        :param A_sparse: sparseness of A matrix
        :param alpha: prior for Dirichlet generating blocks of x
        :return:
        """
        A = (np.random.random((m1, n)) > A_sparse).astype(np.float)
        block_sizes = np.random.multinomial(n,np.ones(m2)/m2)
        x = np.concatenate([np.random.dirichlet(alpha*np.ones(bs)) for bs in \
                            block_sizes])
        b = A.dot(x)
        U = sla.block_diag(*[np.ones(bs) for bs in block_sizes])
        f = U.dot(x)
        scipy.io.savemat(fname, { 'A': A, 'b': b, 'x_true': x, 'U': U, 'f': f },
                         oned_as='column')

    def test_converge_small(self):
        self.generate_data(self.args.file)
        iters, times, states, output = main(args=self.args)
        # Check convergence
        self.assertTrue(output['0.5norm(Ax-b)^2'][-1] < 1e-16)

    def test_converge_x_sparse(self):
        self.generate_data(self.args.file, alpha=0.5)
        iters, times, states, output = main(args=self.args)
        # Check convergence
        self.assertTrue(output['0.5norm(Ax-b)^2'][-1] < 1e-16)

    def test_converge_A_sparse(self):
        self.generate_data(self.args.file, A_sparse=0.01)
        iters, times, states, output = main(args=self.args)
        # Check convergence
        self.assertTrue(output['0.5norm(Ax-b)^2'][-1] < 1e-16)

    def test_converge_medium(self):
        self.generate_data(self.args.file, n=1000, m1=100, m2=10)
        iters, times, states, output = main(args=self.args)
        # Check convergence
        self.assertTrue(output['0.5norm(Ax-b)^2'][-1] < 1e-16)

if __name__ == '__main__':
    unittest.main()
