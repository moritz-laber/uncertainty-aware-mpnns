"""
robustness_radii_realworld.py
----------

Determine robustnes radii on realworld data using Theorem 1.
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

### PARAMETERS ###
data_dir = './data/'                                       # input dir with dataset
model_dir = './models/'                                    # input dir with trained models
input_dir = './experiments/realworld/'                     # input dir with moments
output_dir = './experiments/realworld/'                    # directory to save results

dataset = sys.argv[1]                  # dataset name
num_layers = int(sys.argv[2])          # number of GCN layers
num_hidden = int(sys.argv[3])          # number of hidden units
num_reps = int(sys.argv[4])            # training reps
nonlin_str = sys.argv[5]               # nonlinearity
delta = 0.05                           # misclassification probability

### MAIN ###

## Load Hyperparameters ##
with open(f'{model_dir}{dataset}/hyperparams_model=gcn_data={dataset}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'rb') as f:
    hyperparams = pickle.load(f)

num_features = hyperparams['params']['num_features']
num_nodes = hyperparams['params']['num_nodes']
num_classes = hyperparams['params']['num_classes']
seed = hyperparams['params']['seed']
which_split = hyperparams['params']['which_split']
train_frac = np.round(hyperparams['params']['train_frac'],2)

# select nonlinearity
nonlinearity_dict = {
    'relu' : jax.nn.relu,
    'tanh' : jax.nn.tanh,
    'gelu' : jax.nn.gelu,
    'sigmoid' : jax.nn.sigmoid,
}
nonlinearity = nonlinearity_dict[nonlin_str]

# seeding
# data splits are not needed but required for loading
# similarly weights need to be initialized but are 
# overwritten.
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

## Load Moments ##
with open(f'{input_dir}{dataset}/moments_model=gcn_data={dataset}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'rb') as f:

    moments = pickle.load(f)

## Lipschitzness ##

# Lipschitz constant
start = time.time()
C = gcn_lipschitz(gcn, A_sparse)
end = time.time()
time_C = end - start

# Prediction
start = time.time()
yout = gcn(X, A_sparse, final_embedding=True)
end = time.time()
time_predict = end - start

## Compute robustness radius  ##
results = {}
for estimator, moment_dict in moments['moments'].items():

    if estimator == 'input':
        continue
    
    mu = moment_dict['mu']
    Sig = moment_dict['Sig']

    print(f'Computing robustness for {estimator} estimator and {nonlin_str} nonlinearity.')

    # radius computation 
    mu = mu.reshape(num_nodes, num_classes)
    Sig = Sig.reshape(num_nodes, num_classes, num_nodes, num_classes)

    radii = []
    lengths = []
    start = time.time()
    for v in range(num_nodes):
        
        # get correct and predicted class
        ystar = jnp.argmax(y[v, :]).item()
        ypred = jnp.argmax(yout[v,:]).item()

        if ystar != ypred:
            continue
        else:
            # robustness radius
            ynot = [i for i in range(num_classes) if i!=ystar]
            mu_hat = mu[v, ystar] - mu[v,ynot]
            sig_hat = Sig[v, ystar, v, ystar] + Sig[v, ynot, v, ynot] - 2.0*Sig[v, ystar, v, ynot]
            eps_y = (mu_hat - jnp.sqrt(sig_hat)*jnp.sqrt((1.0 - delta)/delta))
            eps_y /= jnp.sqrt(2.0)*C 
            radii.append(jnp.min(eps_y).item())
            
            # feature vector length
            lengths.append(jnp.linalg.norm(X[v,:]).item())

    lengths = jnp.asarray(lengths)

    # compute certified fraction
    radii = jnp.asarray(radii)
    radii_nan = np.isnan(radii)
    negative_frac = np.sum(radii < 0)/radii.shape[0]
    positive_frac = np.sum(radii > 0)/radii.shape[0]
    zero_frac = np.sum(radii == 0)/radii.shape[0]
    nan_frac = np.sum(np.isnan(radii))/radii.shape[0]
    end = time.time()
    time_radii = end - start
    
    # store in results dict
    results[estimator] = {
            'estimator' : estimator,
            'nonlinearity' : nonlin_str,
            'C_lipschitz' : C,
            'radii' : radii,
            'lengths' : lengths,
            'positive_frac' : positive_frac,
            'negative_frac' : negative_frac,
            'time_radii' : time_radii,
            'time_moment' : moment_dict['time'],
            'time_predict' : time_predict,
            'time_C' : time_C
        }


## Save results ##
with open(f'{output_dir}{dataset}/radii_model=gcn_data={dataset}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'wb') as f:

    pickle.dump(results, f)

