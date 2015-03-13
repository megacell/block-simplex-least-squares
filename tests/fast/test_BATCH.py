import unittest
import numpy as np
from cvxopt import matrix, spdiag, spmatrix, solvers
import sys
sys.path.append('../../')
from python.c_extensions.c_extensions import proj_simplex_c, isotonic_regression_c
from python.algorithm_utils import quad_obj_np, line_search_exact_quad_obj
import python.BATCH as batch
from python.bsls_utils import almost_equal, x2z, qp_to_qp_in_z

__author__ = 'jeromethai'

class TestBatch(unittest.TestCase):
  

    def setUp(self):
        seed = 237423433
        np.random.seed(seed)


    def run_test(self, solver):
        """Use small example of QP on simplex, see:
        http://cvxopt.org/examples/tutorial/qp.html
        """
        Q = 2 * np.array([[2, .5], [.5, 1]])
        c = np.array([1.0, 1.0])
        x_true = np.array([.25, .75])

        def proj(x):
            proj_simplex_c(x, 0, 2)

        def line_search(x, f, g, x_new, f_new, g_new, i):
            return line_search_exact_quad_obj(x, f, g, x_new, f_new, g_new, Q, c)

        def obj(x, g):
            return quad_obj_np(x, Q, c, g)

        x0 = np.array([.5, .5])
        f_min = 1.875

        sol = solver(obj, proj, line_search, x0)
        #print np.linalg.norm(sol['x']-x_true)
        assert almost_equal(sol['x'], x_true)
        assert sol['stop'][-10:] == '< prog_tol'
        sol = solver(obj, proj, line_search, x0, f_min)
        assert almost_equal(sol['x'], x_true)
        assert sol['stop'][-10:] == ' < opt_tol'


    def run_test_2(self, solver):
        # test on constrainted least squares problem
        # min ||Ax-b||^2 = x'A'Ax - 2 b'Ax
        # s.t. ||x||_1 = 1, x>=0
        for i in range(5):
            m = 10 # number of measurements
            n = 7 # dimension of features
            A = np.random.randn(m, n)
            x_true = abs(np.random.randn(n,1))
            x_true = x_true / np.linalg.norm(x_true, 1)
            b = A.dot(x_true)
            Q = A.T.dot(A)
            c = -A.T.dot(b)
            c = c.flatten()
            x0 = np.ones(n) / n

            def proj(x):
                proj_simplex_c(x, 0, n)

            def line_search(x, f, g, x_new, f_new, g_new, i):
                return line_search_exact_quad_obj(x, f, g, x_new, f_new, g_new, Q, c)

            def obj(x, g):
                return quad_obj_np(x, Q, c, g)
                
            x_true = x_true.flatten()
            sol = solver(obj, proj, line_search, x0, prog_tol=1e-14)
            # print np.linalg.norm(sol['x'] - x_true)
            # print sol['stop']
            # print sol['iterations']
            assert almost_equal(sol['x'], x_true, 1e-5)


    def run_test_z(self, solver):
        """Use small example of QP with ordered constraints, see:
        http://cvxopt.org/examples/tutorial/qp.html
        """
        n = 2
        block_starts = np.array([0])
        Q = 2 * np.array([[2, .5], [.5, 1]])
        c = np.array([1.0, 1.0])
        x_true = np.array([.25, .75])
        x_init = np.array([.5, .5])

        # converts into z-variable
        Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
        z_init = x2z(x_init, block_starts=block_starts)

        def obj(x, g):
            return quad_obj_np(x, Qz, cz, g)
        
        def proj(x):
            isotonic_regression_c(x, 0, 1)
            np.minimum(1.,x,x)
            np.maximum(0.,x,x)
            return x

        def line_search(x, f, g, x_new, f_new, g_new, i):
            return line_search_exact_quad_obj(x, f, g, x_new, f_new, g_new, Qz, cz)
        
        # objective in z variable
        f_min = 1.875 - f0

        sol = solver(obj, proj, line_search, z_init)
        assert almost_equal(N.dot(sol['x'])+x0, x_true)
        assert sol['stop'][-10:] == '< prog_tol'
        sol = solver(obj, proj, line_search, z_init, f_min)
        assert almost_equal(N.dot(sol['x'])+x0, x_true)
        assert sol['stop'][-10:] == ' < opt_tol'


    def run_test_z_2(self, solver):
        for i in range(5):
            block_starts = np.array([0])
            m = 10 # number of measurements
            n = 7 # dimension of features
            block_starts = np.array([0])
            A = np.random.randn(m, n)
            x_true = abs(np.random.randn(n,1))
            x_true = x_true / np.linalg.norm(x_true, 1)
            b = A.dot(x_true)
            Q = A.T.dot(A)
            c = -A.T.dot(b)
            c = c.flatten()
            x_init = np.ones(n) / n
            
            # converts into z-variable
            Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
            z_init = x2z(x_init, block_starts=block_starts)

            def obj(x, g):
                return quad_obj_np(x, Qz, cz, g)
        
            def proj(x):
                isotonic_regression_c(x, 0, n-1)
                np.minimum(1.,x,x)
                np.maximum(0.,x,x)
                return x

            def line_search(x, f, g, x_new, f_new, g_new, i):
                return line_search_exact_quad_obj(x, f, g, x_new, f_new, g_new, Qz, cz)
                
            x_true = x_true.flatten()
            sol = solver(obj, proj, line_search, z_init, prog_tol=1e-14)
            assert almost_equal(N.dot(sol['x'])+x0, x_true, 1e-4)



    def test_BATCH(self):
        self.run_test(batch.solve)

    def test_BATCH_2(self):
        self.run_test_2(batch.solve)

    def test_BATCH_BB(self):
        self.run_test(batch.solve_BB)

    def test_BATCH_BB_2(self):
        self.run_test_2(batch.solve_BB)

    def test_BATCH_z(self):
        self.run_test_z(batch.solve)

    def test_BATCH_z_2(self):
        self.run_test_z_2(batch.solve)

    def test_BATCH_BB_z(self):
        self.run_test_z(batch.solve_BB)

    def test_BATCH_BB_z_2(self):
        self.run_test_z_2(batch.solve_BB)

    def test_BATCH_LBFGS(self):
        self.run_test(batch.solve_LBFGS)

    def test_BATCH_LBFGS_2(self):
        self.run_test_2(batch.solve_LBFGS)

if __name__ == '__main__':
    unittest.main()