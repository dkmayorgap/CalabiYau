#
# Quintic_compute_Euler.py
# -------------------
# Computes Euler number from metric.
#

import os, glob
import argparse

import numpy as np
import tensorflow as tf

import sys
sys.path.append("../..")
from models import ProjectiveSpace

def loadRiem(path_to_riems):
    riem = None
    files = sorted(glob.glob(
        os.path.join(path_to_riems, "*.npy")),
        key = lambda x: int(os.path.basename(x).split('.')[0][4:]))
    for riemfile in files:
        if riem is None:
            riem = np.load(riemfile)
        else:
            riem = np.append(riem, np.load(riemfile), axis=0)
    return np.asarray(riem)

def main(args):
    dirname = args.path_to_points

    # Load variables
    data = np.load(os.path.join(dirname, 'dataset.npz'))
    x_vars = tf.cast(tf.constant(data[f"X_{args.mode}"]), tf.float32)
    weights, omegas = data[f'y_{args.mode}'][:,-2], data[f'y_{args.mode}'][:,-1]

    # Load riemann tensors
    riem = loadRiem(args.path_to_riems)

    # Compute C2
    c3 = ProjectiveSpace.getC3(tf.cast(riem, tf.complex128))
    c3_top = ProjectiveSpace.getTopNumber(c3)

    norm_factor = (-2*1j)**3 / np.math.factorial(3)
    print(f"Int_X c3 = {np.real(norm_factor * np.mean(weights * c3_top / omegas))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Computes Euler number for Quintic',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--path_to_points', type=str, help="Path to the directory containing Quintic data.", required=True)
    parser.add_argument('--path_to_riems', type=str, help="Path to the computed riemann tensors", required=True)
    parser.add_argument('--mode', type=str, help="train,val", default='val')
    args = parser.parse_args()
    main(args)

