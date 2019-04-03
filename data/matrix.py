import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

class MaskedMatrix(object):
    def __init__(sampled_elements: np.ndarray, sampled_row_idx: np.ndarray, sampled_col_idx: np.ndarray, m: int, n: int, rank: int) -> None:
        assert (len(array.shape) == 2) 'Currently only functional for 2d matrices'
        assert(rank <= min(m, n)) 'Rank must be less than the smallest dimension of the matrix'
        self.sampled_elements = sampled_elements
        self.sampled_row_idx = sampled_row_idx
        self.sampled_col_idx = sampled_col_idx
        self.rank = rank
        self.m = m
        self.n = n
        self.M_omega = self._create_array()
        self.p = len(sampled_elements)

    def _create_array(self) -> np.ndarray:
        """ Construct a sparse array containing only the entries given
            in `sampled_elements`
        """
        sparse_matrix = sparse.coo_matrix((self.sampled_elements, (self.sampled_row_idx, self.sampled_col_idx)))
        return sparse_matrix.toarray()

    def initialize(self):
        """ Initialize parameters for algorithms
        """
        U, S, Vt = svds(self.array, k=self.rank)
        S = np.diag(S)
        full_matrix = U.dot(S).dot(Vt)
        self.M_constructed = full_matrix
        self.X = U.dot(S)
        self.Y = Vt
        return
