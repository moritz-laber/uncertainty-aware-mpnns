"""
robustness_radii_realworld_cohen.py
----------

Determine robustnes radii on realworld data using
the method of Cohen et al. [1].

[1] Cohen et al. (ICML 2019) Certified Robustness via Randomized Smoothing
https://proceedings.mlr.press/v97/cohen19c.html.
"""

### IMPORTS ###
import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import time

from models import *
from utils import *
from utils_cohen import  *

### PARAMETERS ###
data_dir = './data/'                                       # input dir with dataset
model_dir = './models/'                                    # input dir with trained models
moment_dir = './experiments/realworld/'                    # input dir with moments
output_dir = './experiments/realworld/'                    # directory to save results

dataset = sys.argv[1]                  # dataset name
num_layers = int(sys.argv[2])          # number of GCN layers
num_hidden = int(sys.argv[3])          # number of hidden units
num_reps = int(sys.argv[4])            # training reps
nonlin_str = sys.argv[5]               # nonlinearity

num_samples0 = 100                     # small sample set
num_samples1 = 10_000 - num_samples0   # large sample set

delta = 0.05                           # significance
seed = 1729                            # jax seed for sampling

### MAIN ###

## Seeding ##

## Load Hyperparameters ##
with open(f'{model_dir}{dataset}/hyperparams_model=gcn_data={dataset}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'rb') as f:
    hyperparams = pickle.load(f)

num_features = hyperparams['params']['num_features']
num_nodes = hyperparams['params']['num_nodes']
num_classes = hyperparams['params']['num_classes']
which_split = hyperparams['params']['which_split']
train_frac = np.round(hyperparams['params']['train_frac'],2)

nonlinearity_dict = {
    'relu' : jax.nn.relu,
    'tanh' : jax.nn.tanh,
    'gelu' : jax.nn.gelu,
    'sigmoid' : jax.nn.sigmoid,
}

nonlinearity = nonlinearity_dict[nonlin_str]

# seeding
# only the sampling key is used, we do not need
# train test splits and model weights are loaded.
key = jax.random.PRNGKey(seed=seed)
key, key_model, key_split, key_sampling = jax.random.split(key, 4)

## Load Data ##
data = load_dataset(
    f'{data_dir}{dataset}.npz',
    num_features=num_features,
    which_split=which_split,
    train_frac=train_frac,
    key_split=key_split
    )
A_sparse, X, y, _, _ = data
X = X.reshape(num_nodes, num_features)

## Load Model ##

# initialize model
gcn = GCN(
    key_model,
    dims=[num_features] + num_layers* [num_hidden] + [num_classes],
    non_linearity=nonlinearity,
    final_non_linearity=jax.nn.softmax,
)

# load model parameters
gcn = eqx.tree_deserialise_leaves(
    f'{model_dir}{dataset}/weights_model=gcn_data={dataset}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.eqx',
    gcn
    )

# set standard deviation parameter
with open(f'{moment_dir}{dataset}/moments_model=gcn_data={dataset}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'rb') as f:

    moments = pickle.load(f)

SigX = moments['moments']['input']['Sig']
sigma = jnp.mean(jnp.sqrt(SigX.diagonal()))

## Compute robustness radius & length of the initial feature vector  ##
start = time.time()

# identify correctly predicted nodes
ypred = gcn(X, A_sparse)
correct = jnp.where(y.argmax(axis=1) == ypred.argmax(axis=1))[0]

# predictions and radii
ypred, radii = certify_cohen(
    key_sampling,                    # randomness
    lambda X : gcn(X, A_sparse),     # function 
    sigma,                           # noise standard deviation
    X,                               # features
    (num_samples0, num_samples1),    # estimation, certification samples
    delta,                           # error probability
    True                             # verbose
)

# compute certified fraction
radii = radii[correct]
negative_frac = np.sum(radii < 0)/radii.shape[0]
positive_frac = np.sum(radii > 0)/radii.shape[0]
zero_frac = np.sum(radii == 0)/radii.shape[0]
nan_frac = np.sum(np.isnan(radii))/radii.shape[0]
end = time.time()
time_radii = end - start

# lengths
lengths = jnp.linalg.norm(X, axis=1)
lengths = lengths[correct]

# store in results dict
results = {
        'estimator' : 'cohen',
        'nonlinearity' : nonlin_str,
        'radii' : radii,
        'lengths' : lengths,
        'positive_frac' : positive_frac,
        'negative_frac' : negative_frac,
        'time_radii' : time_radii,
        'num_samples0' : num_samples0,
        'num_samples1' : num_samples1,
        'sigma' : sigma,
        'delta' : delta,
        'seed' : seed
    }

with open(f'{output_dir}{dataset}/cohen_radii_model=gcn_data={dataset}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'wb') as f:

    pickle.dump(results, f)