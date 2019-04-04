import numpy as np
from numpy.linalg import matrix_rank
import argparse
import sys
sys.path.append('..')

from data.matrix import MaskedMatrix
from algorithms.algo import AlternatingSteepestDescent

parser = argparse.ArgumentParser()
parser.add_argument(
    '--m',
    type=int,
    default=100,
    help='number of rows'
)
parser.add_argument(
    '--n',
    type=int,
    default=100,
    help='number of columns'
)
parser.add_argument(
    '--sampling_frac',
    type=float,
    default=0.5,
    help='number of columns'
)

args = parser.parse_args()

# Create the random matrix
M_true = np.random.random(size=(args.m, args.n))
rank = matrix_rank(M_true)

# Get sampled elements and indices from the random matrix
omega_size = int(args.sampling_frac*M_true.size)
flat_index = np.arange(M_true.size)
sampled_flat_indices = np.random.choice(flat_index, size=omega_size, replace=False)
sampled_row_idx, sampled_col_idx = np.unravel_index(sampled_flat_indices, M_true.shape)
sampled_elements = M_true[sampled_row_idx, sampled_col_idx]

# Get the masked matrix
M_masked = MaskedMatrix(sampled_elements=sampled_elements, sampled_row_idx=sampled_row_idx, sampled_col_idx=sampled_col_idx, m=args.m, n=args.n, rank=rank)
M_masked.initialize()

# initialize algorithm
asd = AlternatingSteepestDescent(algo_name='asd')

# optimize
M_new = asd.optimize(M_masked)

print(M_true)
print(M_new.M_constructed)
