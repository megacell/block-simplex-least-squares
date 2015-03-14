import unittest
import numpy as np
from cvxopt import matrix, spdiag, spmatrix, solvers
import sys
sys.path.append('../../')
from python.c_extensions.c_extensions import proj_simplex_c, isotonic_regression_c
from python.algorithm_utils import (quad_obj_np, 
                                    line_search_quad_obj_np,
                                    decreasing_step_size)
import python.BATCH as batch
from python.bsls_utils import (almost_equal, 
                               x2z, 
                               qp_to_qp_in_z,
                               construct_qp_from_least_squares)

__author__ = 'jeromethai'

class TestBatch(unittest.TestCase):
  

    def setUp(self):
        seed = 237423433
        np.random.seed(seed)


    def generate_small_qp(self):
        Q = 2 * np.array([[2, .5], [.5, 1]])
        c = np.array([1.0, 1.0])
        x_true = np.array([.25, .75])       
        w, v = np.linalg.eig(Q) # w[-1] is the smallest eigenvalue
        f_min = 1.875
        min_eig = w[-1]
        return Q, c, x_true, f_min, min_eig


    def generate_random_qp(self, m, n):
        """
        m = # measurements
        n = # dimension of features
        """
        A = np.random.randn(m, n)
        x_true = abs(np.random.randn(n,1))
        x_true = x_true / np.linalg.norm(x_true, 1)
        b = A.dot(x_true)
        x_true = x_true.flatten()
        Q, c = construct_qp_from_least_squares(A, b)
        w, v = np.linalg.eig(Q)
        f_min = quad_obj_np(x_true, Q, c)
        min_eig = w[-1]
        return Q, c, x_true, f_min, min_eig


    def get_solver_parts(self, Q, c, min_eig, in_z = False):
        def step_size(i):
            return decreasing_step_size(i, 1.0, min_eig)

        # if in_z is False:
        def proj(x):
            proj_simplex_c(x, 0, Q.shape[0])
        if in_z:
            def proj(x):
                isotonic_regression_c(x, 0, Q.shape[0])
                np.maximum(0.,x,x)
                np.minimum(1.,x,x)

        def line_search(x, f, g, x_new, f_new, g_new, i):
            return line_search_quad_obj_np(x, f, g, x_new, f_new, g_new, Q, c)

        def obj(x, g=None):
            return quad_obj_np(x, Q, c, g)

        return step_size, proj, line_search, obj


    def test_batch_solver_in_x(self):
        Q, c, x_true, f_min, min_eig = self.generate_small_qp()
        step_size, proj, line_search, obj = self.get_solver_parts(Q, c, min_eig)
        x_init = np.array([.5, .5])
        sol = batch.solve(obj, proj, step_size, x_init)
        assert almost_equal(sol['x'], x_true, 1e-3)
        assert sol['stop'][-10:] == '< prog_tol'
        x_init = np.array([.5, .5])
        sol = batch.solve(obj, proj, step_size, x_init, f_min=f_min)
        assert almost_equal(sol['x'], x_true, 1e-3)
        assert sol['stop'][-10:] == ' < opt_tol'
        x_init = np.array([.5, .5])
        sol = batch.solve(obj, proj, step_size, x_init, line_search)
        assert almost_equal(sol['x'], x_true, 1e-3)


    def test_batch_solver_in_x_2(self):
        for i in range(5):
            n, m = 7, 10
            Q, c, x_true, f_min, min_eig = self.generate_random_qp(m, n)
            step_size, proj, line_search, obj = self.get_solver_parts(Q, c, min_eig)
            x_init = np.ones(n) / n
            sol = batch.solve(obj, proj, step_size, x_init, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-2
            x_init = np.ones(n) / n
            sol = batch.solve(obj, proj, step_size, x_init, line_search, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-2


    def test_batch_solver_in_z(self):
        block_starts = np.array([0])
        Q, c, x_true, f_min, min_eig = self.generate_small_qp()
        Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
        f_min -= f0
        w, v = np.linalg.eig(Qz)
        step_size, proj, line_search, obj = self.get_solver_parts(Qz, cz, w[-1], True)
        x_init = np.array([.5, .5])
        z_init = x2z(x_init, block_starts=block_starts)
        sol = batch.solve(obj, proj, step_size, z_init)
        assert almost_equal(N.dot(sol['x'])+x0, x_true, 1e-3)
        assert sol['stop'][-10:] == '< prog_tol'
        x_init = np.array([.5, .5])
        z_init = x2z(x_init, block_starts=block_starts)
        sol = batch.solve(obj, proj, step_size, z_init, f_min=f_min)
        assert almost_equal(N.dot(sol['x'])+x0, x_true, 1e-3)
        assert sol['stop'][-10:] == ' < opt_tol'
        x_init = np.array([.5, .5])
        z_init = x2z(x_init, block_starts=block_starts)
        sol = batch.solve(obj, proj, step_size, z_init, line_search)
        assert almost_equal(N.dot(sol['x'])+x0, x_true, 1e-3)


    def test_batch_solver_in_z_2(self):
        block_starts = np.array([0])
        for i in range(5):
            n, m = 7, 10
            Q, c, x_true, f_min, min_eig = self.generate_random_qp(m, n)
            Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
            f_min -= f0
            w, v = np.linalg.eig(Qz)
            step_size, proj, line_search, obj = self.get_solver_parts(Qz, cz, w[-1], True)
            x_init = np.ones(n) / n
            z_init = x2z(x_init, block_starts=block_starts)
            sol = batch.solve(obj, proj, step_size, z_init, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-1
            x_init = np.ones(n) / n
            z_init = x2z(x_init, block_starts=block_starts)
            sol = batch.solve(obj, proj, step_size, z_init, line_search, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-1


    def test_bb_solver_in_x(self):
        Q, c, x_true, f_min, min_eig = self.generate_small_qp()
        step_size, proj, line_search, obj = self.get_solver_parts(Q, c, min_eig)
        x_init = np.array([.5, .5])
        sol = batch.solve_BB(obj, proj, line_search, x_init)
        assert almost_equal(sol['x'], x_true, 1e-3)
        assert sol['stop'][-10:] == '< prog_tol'
        x_init = np.array([.5, .5])
        sol = batch.solve_BB(obj, proj, line_search, x_init, f_min=f_min)
        assert almost_equal(sol['x'], x_true, 1e-3)
        assert sol['stop'][-10:] == ' < opt_tol'
        x_init = np.array([.5, .5])
        sol = batch.solve_BB(obj, proj, line_search, x_init)
        assert almost_equal(sol['x'], x_true, 1e-3)


    def test_bb_solver_in_x_2(self):
        for i in range(5):
            n, m = 7, 10
            Q, c, x_true, f_min, min_eig = self.generate_random_qp(m, n)
            step_size, proj, line_search, obj = self.get_solver_parts(Q, c, min_eig)
            x_init = np.ones(n) / n
            sol = batch.solve_BB(obj, proj, line_search, x_init, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-2


    def test_bb_solver_in_z(self):
        block_starts = np.array([0])
        Q, c, x_true, f_min, min_eig = self.generate_small_qp()
        Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
        f_min -= f0
        w, v = np.linalg.eig(Qz)
        step_size, proj, line_search, obj = self.get_solver_parts(Qz, cz, w[-1], True)
        x_init = np.array([.5, .5])
        z_init = x2z(x_init, block_starts=block_starts)
        sol = batch.solve_BB(obj, proj, line_search, z_init)
        assert almost_equal(N.dot(sol['x'])+x0, x_true, 1e-3)
        assert sol['stop'][-10:] == '< prog_tol'
        x_init = np.array([.5, .5])
        z_init = x2z(x_init, block_starts=block_starts)
        sol = batch.solve_BB(obj, proj, line_search, z_init, f_min=f_min)
        assert almost_equal(N.dot(sol['x'])+x0, x_true, 1e-3)
        assert sol['stop'][-10:] == ' < opt_tol'
        x_init = np.array([.5, .5])
        z_init = x2z(x_init, block_starts=block_starts)
        sol = batch.solve_BB(obj, proj, line_search, z_init)
        assert almost_equal(N.dot(sol['x'])+x0, x_true, 1e-3)


    def test_bb_solver_in_z_2(self):
        block_starts = np.array([0])
        for i in range(5):
            n, m = 7, 10
            Q, c, x_true, f_min, min_eig = self.generate_random_qp(m, n)
            Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
            f_min -= f0
            w, v = np.linalg.eig(Qz)
            step_size, proj, line_search, obj = self.get_solver_parts(Qz, cz, w[-1], True)
            x_init = np.ones(n) / n
            z_init = x2z(x_init, block_starts=block_starts)
            sol = batch.solve_BB(obj, proj, line_search, z_init, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-1
            x_init = np.ones(n) / n
            z_init = x2z(x_init, block_starts=block_starts)
            sol = batch.solve_BB(obj, proj, line_search, z_init, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-1



    def test_lbfgs_solver_in_x(self):
        Q, c, x_true, f_min, min_eig = self.generate_small_qp()
        step_size, proj, line_search, obj = self.get_solver_parts(Q, c, min_eig)
        x_init = np.array([.5, .5])
        sol = batch.solve_LBFGS(obj, proj, line_search, x_init)
        assert almost_equal(sol['x'], x_true, 1e-3)
        assert sol['stop'][-10:] == '< prog_tol'
        x_init = np.array([.5, .5])
        sol = batch.solve_LBFGS(obj, proj, line_search, x_init, f_min=f_min)
        assert almost_equal(sol['x'], x_true, 1e-3)
        assert sol['stop'][-10:] == ' < opt_tol'
        x_init = np.array([.5, .5])
        sol = batch.solve_LBFGS(obj, proj, line_search, x_init)
        assert almost_equal(sol['x'], x_true, 1e-3)


    def test_lbfgs_solver_in_x_2(self):
        for i in range(5):
            n, m = 7, 10
            Q, c, x_true, f_min, min_eig = self.generate_random_qp(m, n)
            step_size, proj, line_search, obj = self.get_solver_parts(Q, c, min_eig)
            x_init = np.ones(n) / n
            sol = batch.solve_LBFGS(obj, proj, line_search, x_init, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-2


    def test_lbfgs_solver_in_z(self):
        block_starts = np.array([0])
        Q, c, x_true, f_min, min_eig = self.generate_small_qp()
        Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
        f_min -= f0
        w, v = np.linalg.eig(Qz)
        step_size, proj, line_search, obj = self.get_solver_parts(Qz, cz, w[-1], True)
        x_init = np.array([.5, .5])
        z_init = x2z(x_init, block_starts=block_starts)
        sol = batch.solve_LBFGS(obj, proj, line_search, z_init)
        assert almost_equal(N.dot(sol['x'])+x0, x_true, 1e-3)
        assert sol['stop'][-10:] == '< prog_tol'
        x_init = np.array([.5, .5])
        z_init = x2z(x_init, block_starts=block_starts)
        sol = batch.solve_LBFGS(obj, proj, line_search, z_init, f_min=f_min)
        assert almost_equal(N.dot(sol['x'])+x0, x_true, 1e-3)
        assert sol['stop'][-10:] == ' < opt_tol'
        x_init = np.array([.5, .5])
        z_init = x2z(x_init, block_starts=block_starts)
        sol = batch.solve_LBFGS(obj, proj, line_search, z_init)
        assert almost_equal(N.dot(sol['x'])+x0, x_true, 1e-3)


    def test_lbfgs_solver_in_z_2(self):
        block_starts = np.array([0])
        for i in range(5):
            n, m = 7, 10
            Q, c, x_true, f_min, min_eig = self.generate_random_qp(m, n)
            Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
            f_min -= f0
            w, v = np.linalg.eig(Qz)
            step_size, proj, line_search, obj = self.get_solver_parts(Qz, cz, w[-1], True)
            x_init = np.ones(n) / n
            z_init = x2z(x_init, block_starts=block_starts)
            sol = batch.solve_LBFGS(obj, proj, line_search, z_init, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-1



if __name__ == '__main__':
    unittest.main()