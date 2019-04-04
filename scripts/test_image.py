import argparse
import os
import sys
sys.path.append('..')
import numpy as np
from numpy.linalg import matrix_rank
from PIL import Image

from data.matrix import MaskedMatrix
from algorithms.algo import AlternatingSteepestDescent

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_image',
    type=str,
    default='../examples/images/albert-einstein.jpg',
    help='path to the input image'
)
parser.add_argument(
    '--output_image_dir',
    type=str,
    default='../examples/images/',
    help='directory in which to save reconstructions'
)
parser.add_argument(
    '--sampling_frac',
    type=float,
    default=0.5,
    help='fraction of pixels to keep when randomly sampling'
)
parser.add_argument(
    '--rank',
    type=int,
    default=None,
    help='estimate for the rank of the matrix'
)
parser.add_argument(
    '--max_iterations',
    type=int,
    default=500,
    help='maximum number of algorithm iterations'
)

args = parser.parse_args()

# Load the image as grayscale
img = Image.open(args.input_image).convert('L')
img_array = np.asarray(img).astype('float32')
assert(len(img_array.shape) == 2), 'Image array must be 2d'
m, n = img_array.shape

# Get sampled elements and indices from the random matrix
omega_size = int(args.sampling_frac*img_array.size)
flat_index = np.arange(img_array.size)
sampled_flat_indices = np.random.choice(flat_index, size=omega_size, replace=False)
sampled_row_idx, sampled_col_idx = np.unravel_index(sampled_flat_indices, img_array.shape)
sampled_elements = img_array[sampled_row_idx, sampled_col_idx]

# take a look at the corrupted image
corrupt_img_array = np.zeros(shape=img_array.shape)
corrupt_img_array[sampled_row_idx, sampled_col_idx] = img_array[sampled_row_idx, sampled_col_idx]
corrupt_img = Image.fromarray(corrupt_img_array.astype('uint8'))
corrupt_img_path = os.path.join(args.output_image_dir, 'corrupt_{}_'.format(str(args.sampling_frac))+os.path.basename(args.input_image))
corrupt_img.save(corrupt_img_path)

# Determine heuristic for rank if not provided
if args.rank is None:
    beta = 3.0
    coeff = [1., -(m+n), omega_size/beta]
    roots = np.roots(coeff)
    try:
        rank = int(np.min(roots[roots > 0]))
    except ValueError:
        raise ValueError('Rank calculation is invalid')
else:
    rank = args.rank

# Get the masked matrix
M_masked = MaskedMatrix(sampled_elements=sampled_elements, sampled_row_idx=sampled_row_idx, sampled_col_idx=sampled_col_idx, m=m, n=n, rank=rank)
M_masked.initialize(M_true=img_array)

# initialize algorithm
asd = AlternatingSteepestDescent(algo_name='asd', max_iterations=args.max_iterations)

# optimize
M_new = asd.optimize(M_masked)

constructed_img_array = np.clip(M_new.M_constructed, 0, 255).astype('uint8')
constructed_img = Image.fromarray(constructed_img_array)
constructed_img_path = os.path.join(args.output_image_dir, 'reconstructed_{}_'.format(str(args.sampling_frac))+os.path.basename(args.input_image))
constructed_img.save(constructed_img_path)
