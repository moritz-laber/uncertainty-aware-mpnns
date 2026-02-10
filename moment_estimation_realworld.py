"""
moment_estimation_realworld.py
----------

Estimates the moments of the output distribution
on real-world data and saves them.

M. Laber, 2026/02
"""

### IMPORTS ###
import equinox as eqx
import jax
import numpy as np
import pickle
import time
import sys

from models import *
from utils import *

### PARAMETERS ###

data_dir = './data/'                       # directory with datasets
model_dir = './models/'                    # directory to save trained models
result_dir = './experiments/realworld/'    # directory to save results

dataset = sys.argv[1]                  # dataset name
num_samples = 10000                    # number of samples for moment estimation
noise_level = 0.05                     # input noise level: standard deviation in units of the mean
batch_size = 50                        # batch size for moment estimation
num_layers = int(sys.argv[2])          # number of GCN layers
num_hidden = int(sys.argv[3])          # number of hidden units
num_reps = int(sys.argv[4])            # training reps
nonlin_str = sys.argv[5]               # nonlinearity
beta_relu = 20.0                       # parameter for softplus
eps_taylor = 1e-7                      # parameter for numerical stability in Taylor
seed = 26216                           # jax seed

### MAIN ###

## Seeding ##
# only the feature seed is used, the other two 
# are necessary for model and data loading but
# do not influence results.
key = jax.random.PRNGKey(seed=seed)
key, key_model, key_features, key_split = jax.random.split(key, 4)

## Load Hyperparameters ##
with open(f'{model_dir}{dataset}/hyperparams_model=gcn_data={dataset}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'rb') as f:
    hyperparams = pickle.load(f)

num_features = hyperparams['params']['num_features']
which_split = hyperparams['params']['which_split']
train_frac = np.round(hyperparams['params']['train_frac'])
nonlinearity_dict = {
    'relu' : jax.nn.relu,
    'tanh' : jax.nn.tanh,
    'gelu' : jax.nn.gelu,
    'sigmoid' : jax.nn.sigmoid,
}
nonlinearity = nonlinearity_dict[nonlin_str]

## Load Data ##
data = load_dataset(
    f'{data_dir}{dataset}.npz',
    num_features=num_features,
    which_split=which_split,
    train_frac=train_frac,
    key_split=key_split
    )
A_sparse, X, y, _, _ = data
num_nodes = A_sparse.shape[0]
num_classes = y.shape[1]

## Estimate Input Moments ##
print('Estimating input moments')

# set standard deviation
sigma = noise_level*np.abs(X)

time_start = time.time()
muX, SigX = sample_propagation_batched(
    key_features,
    jax.nn.identity,
    X,
    sigma=sigma,
    num_samples=num_samples,
    batch_size=batch_size,
    )
time_end = time.time()
time_input = time_end - time_start
print(f'Input moment estimation time: {time_input / 60:.4f} minutes')

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

## Moment Propagation ##

# define the exact flattened model
f = flatten_model(
    gcn,
    A_sparse
)

# define the approximate flattened model that replaces ReLU with Softplus
# if nonlinearity is not ReLU use the exact nonlinearity
f_approx = flatten_model(
    gcn,
    A_sparse,
    alternative_nonlinearity=(lambda x: softplus(x, beta=beta_relu)) if nonlinearity==jax.nn.relu else nonlinearity
)

# sampling
print('Moment propagation: sampling')
start = time.time()
muY_sample, SigY_sample = sample_propagation_batched(
    key=key_features,
    f=f,
    mu=X,
    sigma=sigma,
    num_samples=num_samples,
    batch_size=batch_size,
    )
end = time.time()
time_sample = end - start
print(f'Sampling time: {time_sample / 60:.4f} minutes')

# Multivariate Tailor
print('Moment propagation: multivariate Taylor (lin)')
time_start = time.time()
muY_lin, SigY_lin = linear_propagation(
    f_approx,
    muX,
    SigX,
    eps=eps_taylor
    )
time_end = time.time()
time_lin = time_end - time_start
print(f'Multivariate linear Taylor time: {time_lin / 60:.4f} minutes')

print('Moment propagation: multivariate Taylor (quad)')
time_start = time.time()
muY_quad, SigY_quad = quadratic_propagation(
    f_approx,
    muX,
    SigX,
    eps=eps_taylor
    )
time_end = time.time()
time_quad = time_end - time_start
print(f'Multivariate quadratic Taylor time: {time_quad / 60:.4f} minutes')


# PTPE
print('Moment propagation: PTPE')
time_start = time.time()
muY_ptpe, SigY_ptpe = ptpe_gcn(
    gcn,
    A_sparse,
    True,
    muX,
    SigX
    )
time_end = time.time()
time_ptpe = time_end - time_start
print(f'PTPE time: {time_ptpe / 60:.4f} minutes')

# Layerwise Taylor
print('Moment propagation: layerwise Taylor (lin)')
time_start = time.time()
muY_tay1, SigY_tay1 = taylor_gcn(
    gcn,
    A_sparse,
    True,
    muX,
    SigX,
    order=1,
    alternative_nonlinearity=(lambda x: softplus(x, beta=beta_relu)) if nonlinearity==jax.nn.relu else None
    )
time_end = time.time()
time_tay1 = time_end - time_start
print(f'Layerwise linear Taylor time: {time_tay1 / 60:.4f} minutes')

print('Moment propagation: layerwise Taylor (quad)')
time_start = time.time()
muY_tay2, SigY_tay2 = taylor_gcn(
    gcn,
    A_sparse,
    True,
    muX,
    SigX,
    order=2,
    gaussian_closure=False,
    alternative_nonlinearity=(lambda x: softplus(x, beta=beta_relu)) if nonlinearity==jax.nn.relu else None
    )
time_end = time.time()
time_tay2 = time_end - time_start
print(f'Layerwise quadratic Taylor time: {time_tay2 / 60:.4f} minutes')

print('Moment propagation: layerwise Taylor (quad, gc)')
time_start = time.time()
muY_tay2_gc, SigY_tay2_gc = taylor_gcn(
    gcn,
    A_sparse,
    True,
    muX,
    SigX,
    order=2,
    gaussian_closure=True,
    alternative_nonlinearity=(lambda x: softplus(x, beta=beta_relu)) if nonlinearity==jax.nn.relu else None
    )
time_end = time.time()
time_tay2_gc = time_end - time_start
print(f'Layerwise quadratic Taylor with Gaussian closure time: {time_tay2_gc / 60:.4f} minutes')

## Save Results ###
print('Saving results')
result_dict = {'moments': {
                'input' : {'mu': muX, 'Sig': SigX, 'time' : time_input},
                'sample' : {'mu': muY_sample, 'Sig': SigY_sample, 'time' : time_sample},
                'taylor_uni_lin' : {'mu': muY_tay1, 'Sig': SigY_tay1, 'time' : time_tay1},
                'taylor_uni_quad' : {'mu': muY_tay2, 'Sig': SigY_tay2 , 'time' : time_tay2},
                'taylor_uni_quad_gc' : {'mu': muY_tay2_gc, 'Sig': SigY_tay2_gc, 'time' : time_tay2_gc},
                'ptpe' : {'mu': muY_ptpe, 'Sig': SigY_ptpe, 'time' : time_ptpe},
                'taylor_multi_lin' : {'mu': muY_lin, 'Sig': SigY_lin, 'time' : time_lin},
                'taylor_multi_quad' : {'mu': muY_quad, 'Sig': SigY_quad, 'time' : time_quad},
                },  
            'params' : {
                'dataset' : dataset,
                'seed' : seed,
                'which_split' : which_split,
                'num_nodes' : num_nodes,
                'num_features' : num_features,
                'num_classes' : num_classes,
                'noise_level' : noise_level,
                'beta_relu' : beta_relu if nonlinearity==jax.nn.relu else None,
                'eps_taylor' : eps_taylor,
                'nonlinearity' : nonlin_str,
                'num_hidden' : num_hidden,
                'num_layers' : num_layers,
                'num_reps' : num_reps,
                'num_samples' : num_samples,
                'batch_size' : batch_size,
                }
            }

with open(f'{result_dir}{dataset}/moments_model=gcn_data={dataset}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'wb') as f:

    pickle.dump(result_dict, f)



