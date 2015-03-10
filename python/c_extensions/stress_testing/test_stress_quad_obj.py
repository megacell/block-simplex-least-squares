import unittest
import time
import numpy as np
import sys
sys.path.append('../../../')
from python.c_extensions.c_extensions import quad_obj

__author__ = 'jeromethai'

class TestStressQuadObj(unittest.TestCase):

    def setUp(self):
        seed = 237423433
        np.random.seed(seed)
    

    def test_quad_obj(self):
        times_c = []
        times_np = []
        for n in [100, 1000]:
            m = 1.5*n # number of measurements
            A = np.random.randn(m, n)
            x_true = abs(np.random.randn(n))
            x_true = x_true / np.linalg.norm(x_true, 1)
            b = A.dot(x_true)
            Q = A.T.dot(A)
            Q_flat = Q.flatten()
            c = -A.T.dot(b)
            c = c.flatten()
            g = np.zeros(n)

            def obj_c(x, g):
                return quad_obj(x, Q_flat, c, g)

            def obj_np(x, g):
                np.copyto(g, Q.dot(x) + c)
                f = .5 * x.T.dot(g + c)
                return f
            
            start_time = time.time()
            obj_c(x_true, g)
            times_c.append(time.time() - start_time)

            start_time = time.time()
            obj_np(x_true, g)
            times_np.append(time.time() - start_time)
        print 'times for quad_obj in Numpy', times_np
        print 'times for quad_obj in Cython', times_c

if __name__ == '__main__':
    unittest.main()