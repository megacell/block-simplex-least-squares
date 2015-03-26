import unittest
import numpy as np
import sys
sys.path.append('../../')
from python.algorithm_utils import get_solver_parts
import python.BATCH as batch
from python.bsls_utils import (almost_equal, 
                               x2z, 
                               qp_to_qp_in_z,
                               construct_qp_from_least_squares,
                          generate_small_qp,
                            random_least_squares)

__author__ = 'jeromethai'

class TestBatch(unittest.TestCase):
  

    def setUp(self):
        seed = 237423433
        np.random.seed(seed)


    def test_batch_solver_in_x(self):
        block_starts = np.array([0])
        Q, c, x_true, f_min, min_eig = generate_small_qp()
        step_size, proj, line_search, obj = get_solver_parts((Q, c), block_starts, min_eig)
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
        block_starts = np.array([0])
        for i in range(5):
            n, m = 7, 10
            data = random_least_squares(m, n, block_starts)
            Q, c, x_true = data['Q'], data['c'], data['x_true']
            f_min, min_eig = data['f_min'], data['min_eig']
            step_size, proj, line_search, obj = get_solver_parts((Q, c), block_starts, min_eig)
            x_init = np.ones(n) / n
            sol = batch.solve(obj, proj, step_size, x_init, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-2
            x_init = np.ones(n) / n
            sol = batch.solve(obj, proj, step_size, x_init, line_search, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-2


    def test_batch_solver_in_z(self):
        block_starts = np.array([0])
        Q, c, x_true, f_min, min_eig = generate_small_qp()
        Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
        f_min -= f0
        w, v = np.linalg.eig(Qz)
        step_size, proj, line_search, obj = get_solver_parts((Qz, cz), block_starts, w[-1], True)
        x_init = np.array([.5, .5])
        z_init = x2z(x_init, block_starts=block_starts)
        sol = batch.solve(obj, proj, step_size, z_init)
        assert almost_equal(N.dot(sol['x'])+x0, x_true, 1e-3)
        #assert sol['stop'][-10:] == '< prog_tol'
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
            data = random_least_squares(m, n, block_starts)
            Q, c, x_true = data['Q'], data['c'], data['x_true']
            f_min, min_eig = data['f_min'], data['min_eig']
            Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
            f_min -= f0
            w, v = np.linalg.eig(Qz)
            step_size, proj, line_search, obj = get_solver_parts((Qz, cz), block_starts, w[-1], True)
            x_init = np.ones(n) / n
            z_init = x2z(x_init, block_starts=block_starts)
            sol = batch.solve(obj, proj, step_size, z_init, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-1
            x_init = np.ones(n) / n
            z_init = x2z(x_init, block_starts=block_starts)
            sol = batch.solve(obj, proj, step_size, z_init, line_search, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-1


    def test_bb_solver_in_x(self):
        block_starts = np.array([0])
        Q, c, x_true, f_min, min_eig = generate_small_qp()
        step_size, proj, line_search, obj = get_solver_parts((Q, c), block_starts, min_eig)
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
        block_starts = np.array([0])
        for i in range(5):
            n, m = 7, 10
            data = random_least_squares(m, n, block_starts)
            Q, c, x_true = data['Q'], data['c'], data['x_true']
            f_min, min_eig = data['f_min'], data['min_eig']
            step_size, proj, line_search, obj = get_solver_parts((Q, c), block_starts, min_eig)
            x_init = np.ones(n) / n
            sol = batch.solve_BB(obj, proj, line_search, x_init, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-2


    def test_bb_solver_in_z(self):
        block_starts = np.array([0])
        Q, c, x_true, f_min, min_eig = generate_small_qp()
        Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
        f_min -= f0
        w, v = np.linalg.eig(Qz)
        step_size, proj, line_search, obj = get_solver_parts((Qz, cz), block_starts, w[-1], True)
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
            data = random_least_squares(m, n, block_starts)
            Q, c, x_true = data['Q'], data['c'], data['x_true']
            f_min, min_eig = data['f_min'], data['min_eig']
            Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
            f_min -= f0
            w, v = np.linalg.eig(Qz)
            step_size, proj, line_search, obj = get_solver_parts((Qz, cz), block_starts, w[-1], True)
            x_init = np.ones(n) / n
            z_init = x2z(x_init, block_starts=block_starts)
            sol = batch.solve_BB(obj, proj, line_search, z_init, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-1
            x_init = np.ones(n) / n
            z_init = x2z(x_init, block_starts=block_starts)
            sol = batch.solve_BB(obj, proj, line_search, z_init, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-1



    def test_lbfgs_solver_in_x(self):
        block_starts = np.array([0])
        Q, c, x_true, f_min, min_eig = generate_small_qp()
        step_size, proj, line_search, obj = get_solver_parts((Q, c), block_starts, min_eig)
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
        block_starts = np.array([0])
        for i in range(5):
            n, m = 7, 10
            data = random_least_squares(m, n, block_starts)
            Q, c, x_true = data['Q'], data['c'], data['x_true']
            f_min, min_eig = data['f_min'], data['min_eig']
            step_size, proj, line_search, obj = get_solver_parts((Q, c), block_starts, min_eig)
            x_init = np.ones(n) / n
            sol = batch.solve_LBFGS(obj, proj, line_search, x_init, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-2


    def test_lbfgs_solver_in_z(self):
        block_starts = np.array([0])
        Q, c, x_true, f_min, min_eig = generate_small_qp()
        Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
        f_min -= f0
        w, v = np.linalg.eig(Qz)
        step_size, proj, line_search, obj = get_solver_parts((Qz, cz), block_starts, w[-1], True)
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
            data = random_least_squares(m, n, block_starts)
            Q, c, x_true = data['Q'], data['c'], data['x_true']
            f_min, min_eig = data['f_min'], data['min_eig']
            Qz, cz, N, x0, f0 = qp_to_qp_in_z(Q, c, block_starts)
            f_min -= f0
            w, v = np.linalg.eig(Qz)
            step_size, proj, line_search, obj = get_solver_parts((Qz, cz), block_starts, w[-1], True)
            x_init = np.ones(n) / n
            z_init = x2z(x_init, block_starts=block_starts)
            sol = batch.solve_LBFGS(obj, proj, line_search, z_init, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-1


    def test_md_solver_in_x(self):
        block_starts = np.array([0])
        Q, c, x_true, f_min, min_eig = generate_small_qp()
        step_size, proj, line_search, obj = get_solver_parts((Q, c), block_starts, min_eig)
        x_init = np.array([.5, .5])
        sol = batch.solve_MD(obj, block_starts, step_size, x_init)
        assert almost_equal(sol['x'], x_true, 1e-2)
        #assert sol['stop'][-10:] == '< prog_tol'
        x_init = np.array([.5, .5])
        sol = batch.solve_MD(obj, block_starts, step_size, x_init, f_min=f_min)
        assert almost_equal(sol['x'], x_true, 1e-2)
        #assert sol['stop'][-10:] == ' < opt_tol'


    def test_md_solver_in_x_2(self):
        block_starts = np.array([0])
        for i in range(5):
            n, m = 7, 10
            data = random_least_squares(m, n, block_starts)
            Q, c, x_true = data['Q'], data['c'], data['x_true']
            f_min, min_eig = data['f_min'], data['min_eig']
            step_size, proj, line_search, obj = get_solver_parts((Q, c), block_starts, min_eig)
            x_init = np.ones(n) / n
            sol = batch.solve_MD(obj, block_starts, step_size, x_init, f_min=f_min)
            assert obj(sol['x']) - f_min < 1e-2


if __name__ == '__main__':
    unittest.main()