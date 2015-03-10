import unittest
import numpy as np
import sys
sys.path.append('../../')
from python.c_extensions.c_extensions import (proj_simplex_c,
                                       quad_obj, 
                                       line_search_quad_obj)
import python.BATCH as batch
from python.bsls_utils import almost_equal

__author__ = 'jeromethai'

class TestBatch(unittest.TestCase):
  

    def test_batch(self):
        
        Q = 2 * np.array([[2, .5], [.5, 1]])
        c = np.array([1.0, 1.0])
        x_true = np.array([.25, .75])

        def proj(x):
            proj_simplex_c(x, 0, 2)

        def line_search(x, f, g, x_new, f_new, g_new):
            return line_search_quad_obj(x, f, g, x_new, f_new, g_new, Q, c)

        def obj(x, g):
            return quad_obj(x, Q, c, g)

        x0 = np.array([.5, .5])
        f_min = 1.875

        sol = batch.solve(obj, proj, line_search, x0)
        assert almost_equal(sol['x'], x_true)
        assert sol['stop'] == 'stop with f_old-f < prog_tol'
        sol = batch.solve(obj, proj, line_search, x0, f_min)
        assert almost_equal(sol['x'], x_true)
        assert sol['stop'] == 'stop with f-f_min < opt_tol'

if __name__ == '__main__':
    unittest.main()