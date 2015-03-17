import pandas as pd
import unittest
import time
import numpy as np
import sys
import cvxopt as copt

from openopt import QP
sys.path.append('../../../')
from python.c_extensions.c_extensions import (proj_simplex_c,
                                       quad_obj_c,
                                       line_search_quad_obj_c,
                                       isotonic_regression_c)
from python.algorithm_utils import (quad_obj_np,
                                    decreasing_step_size,
                                    get_solver_parts,
                                    save_progress)
import python.BATCH as batch
from python.bsls_utils import (x2z, 
                                qp_to_qp_in_z,
                                random_least_squares,
                                block_starts_to_M,
                                block_starts_to_M2,
                                block_starts_to_N)

__author__ = 'jeromethai'

class TestStressBatch(unittest.TestCase):

    # def setUp(self):
    #     seed = 237423433
    #     seed = 0
    #     seed = 372983
    #     np.random.seed(seed)


    def test_block_starts_to_M(self):
        n = 10
        block_starts = np.array([0, 3, 6])
        M = block_starts_to_M2(block_starts, n)
        print M

        A = (np.random.random((5, 10)) > 0.9).astype(np.float)
        print A

        print A.dot(M)

        # M = block_starts_to_M(block_starts, n, True)
        # print M

        # M = block_starts_to_M(block_starts, n)
        # print M

        # N = block_starts_to_N(block_starts, n, True)
        # print N

        # N = block_starts_to_N(block_starts, n)
        # print N

if __name__ == '__main__':
    unittest.main()
