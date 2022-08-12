#
# Quintic_pointgen.py
# --------------
# Generates points for Quintic
# where the defining polynomial is quintic:
# Q = \sum_{i=0}^4 z_i^5
#

import os
import argparse

import numpy as np
from cymetric.pointgen.pointgen import PointGenerator

def main(args):
    # Define the defining polynomial
    monomials = 5*np.eye(5, dtype=np.int64)
    coefficients = np.ones(5)
    kmoduli = np.ones(1)
    ambient = np.array([4])

    # Initialize point generator
    pg = PointGenerator(monomials, coefficients, kmoduli, ambient)

    dirname = args.output_path
    n_p = args.num_pts
    print(f"Generating {n_p} points to {dirname}...")
    kappa = pg.prepare_dataset(n_p, dirname, val_split=args.val_split)

    # Verify
    data = np.load(os.path.join(dirname, 'dataset.npz'))
    print(f"Generated total of {data['X_train'].shape[0]} training points and {data['X_val'].shape[0]} validation points")

    # Prepare basis
    pg.prepare_basis(dirname, kappa=kappa)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate points for Quintic',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output_path', type=str, help="Path to the output directory for points.", required=True)
    parser.add_argument('--num_pts', type=int, help="Number of points to generate.", default = 100000)
    parser.add_argument('--val_split', type=float, help="Percentage of the points to use for validation.", default = 0.33)

    args = parser.parse_args()
    main(args)