"""
robustness_radii_synthetic_cohen.py
----------

Determine robustnes radii on synthetic data using
the method of Cohen et al. [1].

[1] Cohen et al. (ICML 2019) Certified Robustness via Randomized Smoothing
https://proceedings.mlr.press/v97/cohen19c.html .

M. Laber, 2026/02
"""

### IMPORTS ###
import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse
import numpy as np
import pickle
import sys
import time

from models import *
from utils import *
from utils_cohen import  *

### PARAMETERS ###
data_dir = './data/synthetic/'          # input dir with dataset
model_dir = './models/'                 # input dir with trained models
output_dir = './experiments/synthetic/' # directory to save results

corr = sys.argv[1]                      # correlation
num_layers = int(sys.argv[2])           # number of GCN layers
num_hidden = int(sys.argv[3])           # number of hidden units
num_reps = int(sys.argv[4])             # training reps
nonlin_str = sys.argv[5]                # nonlinearity
graph_num = 0                           # graph number
num_samples0 = 100                      # small sample size
num_samples1 = 10_000 - num_samples0    # large sample size
delta = 0.05                            # significance
seed = 1729                             # jax seed

### MAIN ###

## Seeding ##
# the only randomness is in sampling the 
# model weights are loaded.
key = jax.random.PRNGKey(seed=seed)
key, key_model, key_sampling = jax.random.split(key, 3)

## Load Hyperparameters ##
with open(f'{model_dir}{corr}/hyperparams_model=gcn_corr={corr}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'rb') as f:
    hyperparams = pickle.load(f)

num_features = hyperparams['params']['num_features']
num_nodes = hyperparams['params']['num_nodes']
num_classes = hyperparams['params']['num_classes']

nonlinearity_dict = {
    'relu' : jax.nn.relu,
    'tanh' : jax.nn.tanh,
    'gelu' : jax.nn.gelu,
    'sigmoid' : jax.nn.sigmoid,
}
nonlinearity = nonlinearity_dict[nonlin_str]

## Load Data ##
with open(f'{data_dir}sbm_corr={corr}_graphnum={graph_num}.pkl', 'rb') as f:
    dataset = pickle.load(f)

Adense = dataset['data']['A']
A_sparse = sparse.BCOO.fromdense(Adense)
X = dataset['data']['muX'].reshape(num_nodes, num_features)
y = dataset['data']['y']
SigX = dataset['data']['SigX']

# set standard deviation parameter
sigma = jnp.mean(jnp.sqrt(SigX.diagonal()))

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
    f'{model_dir}{corr}/weights_model=gcn_corr={corr}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.eqx',
    gcn
    )

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

with open(f'{output_dir}{corr}/cohen_radii_model=gcn_corr={corr}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'wb') as f:

    pickle.dump(results, f)