import unittest
import random
import argparse

import numpy as np

from python.main import main
from python.bsls_utils import generate_data

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

    def test_converge_small(self):
        generate_data(fname=self.args.file)
        iters, times, states, output = main(args=self.args)
        # Check convergence
        self.assertTrue(output['0.5norm(Ax-b)^2'][-1] < 1e-16)

    def test_converge_x_sparse(self):
        generate_data(fname=self.args.file, alpha=0.5)
        iters, times, states, output = main(args=self.args)
        # Check convergence
        self.assertTrue(output['0.5norm(Ax-b)^2'][-1] < 1e-16)

    def test_converge_A_sparse(self):
        generate_data(fname=self.args.file, A_sparse=0.05)
        iters, times, states, output = main(args=self.args)
        # Check convergence
        self.assertTrue(output['0.5norm(Ax-b)^2'][-1] < 1e-16)

if __name__ == '__main__':
    unittest.main()
