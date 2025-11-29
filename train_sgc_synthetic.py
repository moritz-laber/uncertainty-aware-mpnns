"""
train_sgc_synthetic.py
----------

Trains SGC model on synthetic datasets using
a given number of layers and iterations.
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
save_result = True                    # whether to save model and hyperparameters
data_dir = './data/synthetic/'        # directory with datasets
result_dir = './models/'              # directory to save trained model
graph_num = 0                         # network index
correlation = sys.argv[1]             # synthetic dataset identified by correlation
num_layers = 3                        # number of sgc layers
num_hidden = 16                       # number of hidden units
num_reps = 1000                       # number of training iterations
eta = 1e-3                            # initial learning rate of AdaBelief
seed = 26216                          # jax seed : 26216

### MAIN SCRIPT ###

## Seeding ##
key = jax.random.PRNGKey(seed=seed)

## Load Data ##
with open(f'{data_dir}sbm_corr={correlation}_graphnum={graph_num}.pkl', 'rb') as f:
    dataset = pickle.load(f)

Adense = dataset['data']['A']
A_sparse = sparse.BCOO.fromdense(Adense)

X = dataset['data']['muX']
y = dataset['data']['y']
train_idx = dataset['data']['train_idx']
test_idx = dataset['data']['test_idx']

num_classes = dataset['params']['num_classes']
num_nodes = dataset['params']['num_nodes']
num_features = dataset['params']['num_features']

## Train Model ##

# initialize model
key, key_ = jax.random.split(key)
sgc = SGC(
    key_,
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
    (A_sparse, Xdata, y),
    train_idx,
    num_reps=num_reps
)

# evaluate model
test_acc = accuracy(sgc, (A_sparse, Xdata, y), test_idx)
train_acc = accuracy(sgc, (A_sparse, Xdata, y), train_idx)
print(f'Test accuracy: {test_acc:.4f}, Train accuracy: {train_acc:.4f}')

# save trained model
if save_result:
    eqx.tree_serialise_leaves(f'{result_dir}{correlation}/weights_model=sgc_corr={correlation}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.eqx', sgc)

# save parameters
param_dict = {'params' : {
                'correlation' : correlation,
                'model' : 'sgc',
                'seed' : seed,
                'num_nodes' : num_nodes,
                'num_features' : num_features,
                'num_classes' : num_classes,
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
    with open(f'{result_dir}{correlation}/hyperparams_model=sgc_corr={correlation}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'wb') as f:

        pickle.dump(param_dict, f)



