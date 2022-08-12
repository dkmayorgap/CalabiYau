#
# Quintic_calc_riem.py
# --------------
# Calculates values of riemann tensor
# at specified points.
#

import time

import os, glob
import argparse

import numpy as np
import tensorflow as tf

from cymetric.models.tfmodels import PhiFSModel
from cymetric.models.tfhelper import prepare_tf_basis


import sys
sys.path.append("../..")
from models import ProjectiveSpace

def main(args):
    dirname = args.path_to_points

    data = np.load(os.path.join(dirname, 'dataset.npz'))
    BASIS = np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True)
    BASIS = prepare_tf_basis(BASIS)

    # Set up NN
    nn_phi = tf.keras.models.load_model(args.path_to_model)
    pfs_model = PhiFSModel(nn_phi, BASIS)

    # Load variables
    x_vars = tf.cast(tf.constant(data[f"X_{args.mode}"]), tf.float32)
    weights, omegas = data[f'y_{args.mode}'][:,-2], data[f'y_{args.mode}'][:,-1]

    print(f"Loaded total of {x_vars.shape[0]} points.")

    print(np.mean(weights * np.linalg.det(pfs_model(x_vars, training=False))/ omegas))
    return

    batch_size = args.batch_size
    rem_pts = x_vars.shape[0] % batch_size
    num_partitions = ((x_vars.shape[0] - rem_pts) // batch_size) + (1 if rem_pts > 0 else 0)

    continue_from = 0
    if len(glob.glob(os.path.join(args.path_to_output, "*.npy"))) > 0:
        continue_from = max(
            map(lambda x: int(os.path.basename(x).split('.')[0][4:]),
                glob.glob(
                    os.path.join(args.path_to_output, "*.npy")))) + 1
    for i in range(continue_from, num_partitions):
        a = batch_size*i
        b = min(batch_size*(i+1), x_vars.shape[0])

        start_time = time.time()

        print(f"Generating {a}:{b} in {i+1}/{num_partitions}")
        riem = ProjectiveSpace.getRiemannPB(
            x_vars[a:b], pfs_model.fubini_study_pb if args.metric == "fs" else lambda x: pfs_model(x, training=False),
            pfs_model.pullbacks, 5)
        np.save(os.path.join(args.path_to_output, f"riem{i}"), riem)

        end_time = time.time()
        print("Took: {:.02f}".format(end_time - start_time))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute Riemann tensor values for Quintic',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--path_to_points', type=str, help="Path to the directory containing Quintic data.", required=True)
    parser.add_argument('--path_to_model', type=str, help="Path to the TensorFlow model.", required=True)
    parser.add_argument('--path_to_output', type=str, help="Path to the output directory.", required=True)
    parser.add_argument('--batch_size', type=str, help="Number of points per computation.", default=500)
    parser.add_argument('--metric', type=str, help="fs,pred", default="pred")
    parser.add_argument('--mode', type=str, help="train,val", default='train')
    args = parser.parse_args()
    main(args)
