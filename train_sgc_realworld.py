"""
train_sgc_realworld.py
----------

Trains a SGC model with a given number of layers with
a given number of iterations.

M. Laber, 2026/02
"""

### IMPORTS ###
import equinox as eqx
import jax
import optax
import pickle

from models import *
from utils import *

import sys

### PARAMETERS ###
save_result = True              # whether to save model and hyperparameters
inductive = True                # whether to use an inductive split
data_dir = './data/'            # directory with datasets
result_dir = './models/'        # directory to save trained model
dataset = 'pubmed'              # dataset name
which_split = 0                 # which data split to use if multiple are available
train_frac = 0.25               # fraction of nodes used for training if no split is provided
num_features = 5                # number of node features
num_layers = int(sys.argv[1])   # number of GCN layers
num_reps = int(sys.argv[2])     # number of training iterations
eta = 1e-3                      # initial learning rate of AdaBelief
seed = 26216                    # jax seed : 26216

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
    key_split=key_split,
    inductive=inductive
    )

if inductive:
    A_train, A_test, X, y, train_idx, test_idx = data # type: ignore
else:
    A_train, X, y, train_idx, test_idx = data # type: ignore
    A_test = A_train

num_nodes = A_train.shape[0]
num_classes = y.shape[1]

## Train Model ##

# initialize model
sgc = SGC(
    key_model,
    n_layers=num_layers,
    n_features=num_features,
    n_classes=num_classes,
    non_linearity=jax.nn.softmax
)

# train model
Xdata = X.reshape(num_nodes, num_features)
sgc, sgc_loss_list = train(
    sgc,
    optax.adabelief(eta),
    (A_train, Xdata, y),
    train_idx,
    num_reps=num_reps
)

# evaluate model
test_acc = accuracy(sgc, (A_test, Xdata, y), test_idx)
train_acc = accuracy(sgc, (A_train, Xdata, y), train_idx)
print(f'Test accuracy: {test_acc:.4f}, Train accuracy: {train_acc:.4f}')

# save trained model
if save_result:
    eqx.tree_serialise_leaves(f'{result_dir}{dataset}/weights_model=sgc_data={dataset}_l={num_layers}_reps={num_reps}.eqx', sgc)

# save parameters
param_dict = {'params' : {
                'dataset' : dataset,
                'model' : 'sgc',
                'seed' : seed,
                'which_split' : which_split,
                'num_nodes' : num_nodes,
                'num_features' : num_features,
                'num_classes' : num_classes,
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
    with open(f'{result_dir}{dataset}/hyperparams_model=sgc_data={dataset}_l={num_layers}_reps={num_reps}.pkl', 'wb') as f:

        pickle.dump(param_dict, f)
