"""
train_gcn_realworld.py
----------

Trains GCN model on real-world datasets for a given number of layers
and hidden units iterating over relu, gelu, tanh, sigmoid nonlinearities.
"""

### IMPORTS ###
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse
import numpy as np
import optax
import pickle
import time

from models import *
from utils import *

import sys

### PARAMETERS ###
save_result = True                         # whether to save model and hyperparameters
data_dir = './data/'                       # directory with datasets
result_dir = './models/'                   # directory to save trained model
dataset = 'cornell'                        # dataset name
which_split = 0                            # which data split to use if multiple are available
train_frac = 0.25                          # fraction of nodes used for training if no split is provided
num_features = 5                           # number of node features: 5 (None to use all)
num_layers = int(sys.argv[1])              # number of GCN layers: 2, 3
num_hidden = int(sys.argv[2])              # number of hidden units: 4, 8
nonlinearities = [                         # nonlinearities to consider, (string, function) pairs
    ('relu' , jax.nn.relu),  
    ('tanh',  jax.nn.tanh),
    ('gelu', jax.nn.gelu),
    ('sigmoid', jax.nn.sigmoid)
    ]
num_reps = int(sys.argv[3])      # number of training iterations: 250, 500, 1000, 2000, 4000
eta = 1e-3                       # initial learning rate of AdaBelief
seed = 26216                     # jax seed: 26216

### MAIN SCRIPT ###

## Seeding ##
key = jax.random.PRNGKey(seed=seed)
key, key_model, key_split = jax.random.split(key, 3)

## Load Data ##
data = load_dataset(
    f'{data_dir}{dataset}.npz',
    num_features=num_features,
    which_split=which_split,
    train_frac=train_frac,
    key_split=key_split
    )
A_sparse, X, y, train_idx, test_idx = data
num_nodes = A_sparse.shape[0]
num_classes = y.shape[1]

for nonlin_str, nonlinearity in nonlinearities:

    ## Train Model ##
    print(f'Training on {dataset} with l={num_layers} layers, h={num_hidden} hidden units, and {nonlin_str} nonlinearity')

    # initialize model
    key, key_ = jax.random.split(key)
    gcn = GCN(
        key_,
        dims=[num_features] + num_layers* [num_hidden] + [num_classes],
        non_linearity=nonlinearity,
        final_non_linearity=jax.nn.softmax,
    )

    # train model
    Xdata = X.reshape(num_nodes, num_features)
    gcn, gcn_loss_list = train(
        gcn,
        optax.adabelief(eta),
        (A_sparse, Xdata, y),
        train_idx,
        num_reps=num_reps
    )

    # evaluate model
    test_acc = accuracy(gcn, (A_sparse, Xdata, y), test_idx)
    train_acc = accuracy(gcn, (A_sparse, Xdata, y), train_idx)
    print(f'Test accuracy: {test_acc:.4f}, Train accuracy: {train_acc:.4f}')

    # save trained model
    if save_result:
        eqx.tree_serialise_leaves(f'{result_dir}{dataset}/weights_model=gcn_data={dataset}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.eqx', gcn)
    
    # save parameters
    param_dict = {'params' : {
                    'dataset' : dataset,
                    'model' : 'gcn',
                    'seed' : seed,
                    'which_split' : which_split,
                    'num_nodes' : num_nodes,
                    'num_features' : num_features,
                    'num_classes' : num_classes,
                    'nonlinearity' : nonlin_str,
                    'num_hidden' : num_hidden,
                    'num_layers' : num_layers,
                    'train_frac' : float(train_idx.shape[0])/num_nodes,
                    'test_frac' : float(test_idx.shape[0])/num_nodes,
                    'test_acc' : test_acc,
                    'train_acc' : train_acc,
                    'learning_rate' : eta,
                    'num_reps' : num_reps,
                    }
                }
    
    if save_result:
        with open(f'{result_dir}{dataset}/hyperparams_model=gcn_data={dataset}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'wb') as f:
    
            pickle.dump(param_dict, f)