"""
moment_estimation_synthetic.py
----------

Estimates the moments of the output distribution
on synthetic data and saves them.
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
import sys

from models import *
from utils import *

### PARAMETERS ###
data_dir = './data/synthetic/'             # directory with datasets
model_dir = './models/'                    # directory to save trained models
result_dir = './experiments/synthetic/'    # directory to save results

correlation = sys.argv[1]              # correlation type
num_layers = int(sys.argv[2])          # number of GCN layers
num_hidden = int(sys.argv[3])          # number of hidden units
num_reps = int(sys.argv[4])            # training reps
nonlin_str = sys.argv[5]               # nonlinearity
graphnum = 0                           # graph number
beta_relu = 20.0                       # parameter for softplus
eps_taylor = 1e-7                      # parameter for numerical stability in Taylor
seed = 26216                           # jax seed : dummy not used.

### MAIN ###

## Seeding ##
# seed is necessary to initialize the model but
# they are later on overwritten with loaded weights.
key = jax.random.PRNGKey(seed=seed)
key, key_model = jax.random.split(key, 2)

## Load Data & Parameters ##
with open(f'{data_dir}/sbm_corr={correlation}_graphnum={graphnum}.pkl', 'rb') as f:
    data = pickle.load(f)

num_features = data['params']['num_features']
num_nodes = data['params']['num_nodes']
num_classes = data['params']['num_classes']
num_samples = data['params']['num_samples']
muX = data['data']['muX']
SigX = data['data']['SigX']
X = data['data']['Xsamples']
num_samples = X.shape[-1]
Adense = data['data']['A']
A_sparse = sparse.BCOO.fromdense(Adense)

nonlinearity_dict = {
    'relu' : jax.nn.relu,
    'tanh' : jax.nn.tanh,
    'gelu' : jax.nn.gelu,
    'sigmoid' : jax.nn.sigmoid,
}
nonlinearity = nonlinearity_dict[nonlin_str]


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
    f'{model_dir}{correlation}/weights_model=gcn_corr={correlation}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.eqx',
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
muY_sample, SigY_sample = sample_propagation(
        f,
        X
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
                'sample' : {'mu': muY_sample, 'Sig': SigY_sample, 'time' : time_sample},
                'taylor_uni_lin' : {'mu': muY_tay1, 'Sig': SigY_tay1, 'time' : time_tay1},
                'taylor_uni_quad' : {'mu': muY_tay2, 'Sig': SigY_tay2 , 'time' : time_tay2},
                'taylor_uni_quad_gc' : {'mu': muY_tay2_gc, 'Sig': SigY_tay2_gc, 'time' : time_tay2_gc},
                'ptpe' : {'mu': muY_ptpe, 'Sig': SigY_ptpe, 'time' : time_ptpe},
                'taylor_multi_lin' : {'mu': muY_lin, 'Sig': SigY_lin, 'time' : time_lin},
                'taylor_multi_quad' : {'mu': muY_quad, 'Sig': SigY_quad, 'time' : time_quad},
                },  
            'params' : {
                'correlation' : correlation,
                'seed' : seed,
                'num_nodes' : num_nodes,
                'num_features' : num_features,
                'num_classes' : num_classes,
                'beta_relu' : beta_relu if nonlinearity==jax.nn.relu else None,
                'eps_taylor' : eps_taylor,
                'nonlinearity' : nonlin_str,
                'num_hidden' : num_hidden,
                'num_layers' : num_layers,
                'num_reps' : num_reps,
                'num_samples' : num_samples
                }
            }

with open(f'{result_dir}{correlation}/moments_model=gcn_corr={correlation}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'wb') as f:
    
    pickle.dump(result_dict, f)



