

__author__ = 'jeromethai'


class TestSparseGradient(unittest.TestCase):

    def test_sparse_gradient(self):

        for i,n in enumerate([10, 100, 1000]):
            m1 = n/4
            A_sparse = 0.9
            data = generate_data(n=n, m1=m1, A_sparse=A_sparse)
            A, b, x_true = data['A'], data['b'], data['x_true']