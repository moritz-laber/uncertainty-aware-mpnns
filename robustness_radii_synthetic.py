"""
robustness_radii_synthetic.py
----------

Determine robustnes radii on synthetic data using Theorem 1.
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
data_dir = './data/synthetic/'                # input dir with dataset
model_dir = './models/'                       # input dir with trained models
input_dir = './experiments/synthetic/'        # input dir with moments
output_dir = './experiments/synthetic/'       # directory to save results

corr = sys.argv[1]                     # correlation name
num_layers = int(sys.argv[2])          # number of GCN layers
num_hidden = int(sys.argv[3])          # number of hidden units
num_reps = int(sys.argv[4])            # training reps
nonlin_str = sys.argv[5]               # nonlinearity
graph_num = 0                          # graph number among synthetic graphs
delta = 0.05                           # misclassification probability

### MAIN ###

## Seeding ##
key = jax.random.PRNGKey(seed=10)
key, key_model = jax.random.split(key, 2) # key for model initialization but weights are loaded.

## Load Hyperparameters ##
with open(f'{model_dir}{corr}/hyperparams_model=gcn_corr={corr}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'rb') as f:
    hyperparams = pickle.load(f)

num_features = hyperparams['params']['num_features']
num_nodes = hyperparams['params']['num_nodes']
num_classes = hyperparams['params']['num_classes']

# set the nonlinearity
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

## Loading ##

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

# load moments
with open(f'{input_dir}{corr}/moments_model=gcn_corr={corr}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'rb') as f:

    moments = pickle.load(f)

## Lipschitzness ##

# Lipschitz constant
start = time.time()
C = gcn_lipschitz(gcn, A_sparse)
end = time.time()
time_C = end - start

# Predict labels
start = time.time()
yout = gcn(X, A_sparse, final_embedding=True)
end = time.time()
time_predict = end - start

## Compute robustness radius & length of the initial feature vector  ##
results = {}
for estimator, moment_dict in moments['moments'].items():

    if estimator == 'intput':
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
            sig_hat = Sig[v, ystar, v, ystar] + Sig[v, ynot, v, ynot] - 2*Sig[v, ystar, v, ynot]
            eps_y = mu_hat - jnp.sqrt(sig_hat)*jnp.sqrt((1.0 - delta)/delta)
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
            'estimator' : estimator ,
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
with open(f'{output_dir}{corr}/radii_model=gcn_corr={corr}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'wb') as f:

    pickle.dump(results, f)