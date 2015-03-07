import unittest
import random
import argparse

import numpy as np
import scipy.linalg as sla
import scipy.io

from python.main import main
from python.util import generate_data

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

    def test_converge_medium(self):
        generate_data(self.args.file, n=1000, m1=100, m2=10)
        iters, times, states, output = main(args=self.args)
        # Check convergence
        self.assertTrue(output['0.5norm(Ax-b)^2'][-1] < 1e-16)

if __name__ == '__main__':
    unittest.main()
