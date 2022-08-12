#
# Quintic_train.py
# --------------
# Trains Quintic model.
#

import os
import json
import argparse

import numpy as np
import tensorflow as tf
tfk = tf.keras

from cymetric.models.tfhelper import prepare_tf_basis, train_model
from cymetric.models.tfmodels import PhiFSModel
from cymetric.models.callbacks import RicciCallback, SigmaCallback, VolkCallback, KaehlerCallback, TransitionCallback
from cymetric.models.metrics import SigmaLoss, KaehlerLoss, TransitionLoss, VolkLoss, RicciLoss, TotalLoss


def loadCallbacks(data):
    rcb = RicciCallback((data['X_val'], data['y_val']), data['val_pullbacks'])
    scb = SigmaCallback((data['X_val'], data['y_val']))
    volkcb = VolkCallback((data['X_val'], data['y_val']))
    kcb = KaehlerCallback((data['X_val'], data['y_val']))
    tcb = TransitionCallback((data['X_val'], data['y_val']))
    cb_list = [rcb, scb, kcb, tcb, volkcb]
    return cb_list

def loadMetrics():
    cmetrics = [TotalLoss(), SigmaLoss(), KaehlerLoss(), TransitionLoss(), VolkLoss(), RicciLoss()]
    return cmetrics


def createModel(model_specification):
    nlayer = model_specification['nlayer']
    nHidden = model_specification['nHidden']
    act = model_specification['act']
    n_in = model_specification['n_in']
    n_out = model_specification['n_out']

    # Set up NN
    nn_phi = tf.keras.Sequential()
    nn_phi.add(tfk.Input(shape=(n_in)))
    for i in range(nlayer):
        nn_phi.add(tfk.layers.Dense(nHidden, activation=act))
    nn_phi.add(tfk.layers.Dense(n_out, use_bias=False))

    return nn_phi


def loadModelSpecification(path):
    with open(path, 'r') as f:
        return json.load(f)

def main(args):
    dirname = args.path_to_points

    data = np.load(os.path.join(dirname, 'dataset.npz'))
    BASIS = np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True)
    BASIS = prepare_tf_basis(BASIS)

    print(f"Loaded total of {data['X_train'].shape[0]} training points and {data['X_val'].shape[0]} validation points.")

    model_specification = loadModelSpecification(args.path_to_model_specification)
    print("Loaded model specification:")
    print(model_specification)
    print()

    # Load model
    nn_phi = createModel(model_specification)
    alpha = model_specification['alpha']
    phimodel = PhiFSModel(nn_phi, BASIS, alpha=alpha)

    # Initialize training
    opt_phi = tfk.optimizers.Adam()
    cb_list = loadCallbacks(data)
    cmetrics = loadMetrics()

    # Start training
    phimodel, training_history = train_model(
        phimodel, data, optimizer=opt_phi,
        epochs=model_specification['nEpochs'], batch_sizes=model_specification['bSizes'], verbose=1,
        custom_metrics=cmetrics, callbacks=cb_list)

    # Save the model
    tf.keras.models.save_model(nn_phi, args.output_directory)
    np.save(os.path.join(args.output_directory, "training_history"), training_history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train PhiFSModel for Quintic',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--path_to_points', type=str, help="Path to the directory containing Quintic data.", required=True)
    parser.add_argument('--output_directory', type=str, help="Path to the output directory for the model.", required=True)
    parser.add_argument('--path_to_model_specification', type=str, help="Path to the model specification json.", default="model_specification.json")

    args = parser.parse_args()
    main(args)
