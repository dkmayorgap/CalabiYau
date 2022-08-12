import time

import os, glob
import argparse

import numpy as np
import tensorflow as tf
import gc

from cymetric.models.tfmodels import PhiFSModel
from cymetric.models.tfhelper import prepare_tf_basis

import sys
sys.path.append("../..")
from models import ProjectiveSpace

dirname = './QuinticData'#args.dir_name
mode = 'val'#args.mode

data = np.load(os.path.join(dirname, 'dataset.npz'))
BASIS = np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True)
BASIS = prepare_tf_basis(BASIS)

    # Set up NN
nn_phi = tf.keras.models.load_model('QuinticModel')
pfs_model = PhiFSModel(nn_phi, BASIS)

    # Load variables
x_vars = tf.cast(tf.constant(data[f"X_{mode}"]), tf.float32)
weights, omegas = data[f'y_{mode}'][:,-2], data[f'y_{mode}'][:,-1]

#print(f"Loaded total of {x_vars.shape[0]} points.")
#print(np.mean(weights * np.linalg.det(pfs_model(x_vars, training=False))/ omegas))

batch_size = 250 #args.batch_size
rem_pts = x_vars.shape[0] % batch_size
num_partitions = ((x_vars.shape[0] - rem_pts) // batch_size) + (1 if rem_pts > 0 else 0)

path_to_output = 'riemvaluesfs'
ametric = 'fs'

continue_from = 0
if len(glob.glob(os.path.join(path_to_output, "*.npy"))) > 0:
    continue_from = max(
        map(lambda x: int(os.path.basename(x).split('.')[0][4:]),
            glob.glob(
                os.path.join(path_to_output, "*.npy")))) + 1
i = continue_from
if i < num_partitions:
    a = batch_size*i
    b = min(batch_size*(i+1), x_vars.shape[0])

    start_time = time.time()

    print(f"Generating {a}:{b} in {i+1}/{num_partitions}")
    riem = ProjectiveSpace.getRiemannPB(
        x_vars[a:b], pfs_model.fubini_study_pb if ametric == "fs" else lambda x: pfs_model(x, training=False),
        pfs_model.pullbacks, 5)
    np.save(os.path.join(path_to_output, f"riem{i}"), riem)

    end_time = time.time()
    print("Took: {:.02f}".format(end_time - start_time))
    print()
    gc.collect(generation=2)