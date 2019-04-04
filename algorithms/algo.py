from abc import ABC, abstractmethod
import sys
sys.path.append('..')
from data.matrix import MaskedMatrix
import numpy as np
from numpy.linalg import norm
import time

class CompletionAlgorithm(ABC):
    def __init__(self, algo_name, **kwargs):
        self.params = {
            'max_iterations': 500,
            'tol_res': 1e-5,
            'tol_chg': 1e-5,
            'tol_rel': 1-(1e-5),
            'shift_rel': 15,
            'algo_name': algo_name
            }
        self.params.update(kwargs)
        self.stats = {
            'err_res': np.zeros(self.params['max_iterations']),
            'err_rel': np.zeros(self.params['max_iterations']),
            'norm_M': np.zeros(self.params['max_iterations']),
            'err_M': np.zeros(self.params['max_iterations']),
        }

    @abstractmethod
    def optimize(self, data):
        raise NotImplementedError("Should be implemented by subclassing")


class AlternatingSteepestDescent(CompletionAlgorithm):
    def optimize(self, matrix: MaskedMatrix):
        """ Loop through the iteration process to create the fully
            constructed matrix
        """
        k = 0
        temp_X = np.zeros(shape=(matrix.m, matrix.n))
        temp_Y = np.zeros(shape=(matrix.m, matrix.n))
        self._calculate_stats(matrix, k)
        # iterate through steepest descent (we've already taken the first step)
        k = 1
        condition = self._update_condition(k)
        start_time = time.time()
        while condition:
            # Step 1 in Algorithm 3: calculate gradient for X
            res = matrix.sampled_elements - matrix.M_constructed[matrix.sampled_row_idx, matrix.sampled_col_idx]
            temp_X[matrix.sampled_row_idx, matrix.sampled_col_idx] = res.copy()
            grad_X = -temp_X.dot(matrix.Y.transpose())
            temp_tx = grad_X.dot(matrix.Y)
            tx = norm(grad_X, 'fro')**2 / norm(temp_tx[matrix.sampled_row_idx, matrix.sampled_col_idx], 2)**2

            # Step 2 in Algorithm 3: update X (and M)
            matrix.X = matrix.X - tx*(grad_X)
            matrix.M_constructed = (matrix.X).dot(matrix.Y)

            # Step 3 in Algorithm 3: calculate gradient for Y
            res = matrix.sampled_elements - matrix.M_constructed[matrix.sampled_row_idx, matrix.sampled_col_idx]
            temp_Y[matrix.sampled_row_idx, matrix.sampled_col_idx] = res.copy()
            grad_Y = -matrix.X.transpose().dot(temp_Y)
            temp_ty = matrix.X.dot(grad_Y)
            ty = norm(grad_Y, 'fro')**2 / norm(temp_ty[matrix.sampled_row_idx, matrix.sampled_col_idx], 2)**2

            # Step 4 in Algorithm 3: update Y (and M)
            matrix.Y = matrix.Y - ty*(grad_Y)
            matrix.M_constructed = (matrix.X).dot(matrix.Y)

            self._calculate_stats(matrix, k)
            condition = self._update_condition(k)

            end_time = time.time()
            print('Iteration: {}'.format(k))
            stats = [val[k] for key, val in self.stats.items()]
            print('Stats: {}'.format(stats))
            k += 1

        print('Finished in {} seconds'.format(end_time-start_time))

        return matrix


    def _update_condition(self, k):
        """ Update stopping criterion at each iteration k
        """
        cond1 = (k <= self.params['max_iterations'])
        cond2 = (self.stats['err_res'][k] > self.params['tol_res'])
        cond3 = (self.stats['err_rel'][k] < self.params['tol_rel'])
        print([x for x in [cond1, cond2, cond3]])
        condition = ((cond1 and cond2 and cond3) or k < 2)
        return condition

    def _calculate_stats(self, matrix, k):
        """ Update stats dictionary at end of iteration k
        """
        observed_norm = norm(matrix.sampled_elements, 2)
        res = matrix.sampled_elements - matrix.M_constructed[matrix.sampled_row_idx, matrix.sampled_col_idx]
        res_norm = norm(res, 2)
        err_res = res_norm / observed_norm
        self.stats['err_res'][k] = err_res
        self.stats['norm_M'][k] = norm(matrix.M_constructed, 'fro')
        shift = self.params['shift_rel']
        if k > shift + 1:
            self.stats['err_rel'][k] = err_res / self.stats['err_rel'][k-shift]**(1./shift)
        else:
            self.stats['err_rel'][k] = 0.
        return
