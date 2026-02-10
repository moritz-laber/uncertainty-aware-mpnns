"""
create_dataset.py
----------

Generator for the synthetic datasets: Graphs are
sampled from SBM with two groups and features are
samples from a Gaussian Markov Random Field.

M. Laber, 2026/02
"""

### IMPORTS ###
import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse
import numpy as np
import pickle
import sys

from models import *
from utils import *

### PARAMETERS ###
data_dir = './data/'    # directory to save results
num_networks = 6        # number of networks
num_nodes = 256         # number of nodes
num_features = 3        # number of node features
num_classes = 2         # number of classes
num_samples = 10000     # number of samples for moment estimation
train_frac = 0.7        # training set fraction

ns = [                  # community sizes
    num_nodes//2,
    num_nodes - num_nodes//2
    ]

k_ii = 8                # average number of intra community neighbors
k_ij = 4                # average number of inter community neighbors

mu0 = -2.0 * jnp.ones(num_features)       # mean for class 0
mu1 =  2.0 * jnp.ones(num_features)       # mean for class 1

correlation = sys.argv[1]       # correlation type: inif, indf, dndf

if correlation=='inif':
    sig_node = 0.25
    rho_node = 0.00
    beta = 0.00
elif correlation=='indf':
    sig_node = 0.25
    rho_node = 0.50
    beta = 0.00
elif correlation=='dndf':
    sig_node = 1.00
    rho_node = 0.50
    beta = 1.50
else:
    raise ValueError('Unknown Correlation!')

seed_jax = 26216    # jax seed
seed_numpy = 1124   # numpy seed (for SBM)

### MAIN ###

# seeding
rng = np.random.default_rng(seed_numpy)
key = jax.random.PRNGKey(seed=seed_jax)

for i in range(num_networks):
    print(f'Creating network {i+1}/{num_networks}.')

    # split key
    key, key_features, key_split = jax.random.split(key, 3)

    # sample graph from SBM
    A, g = sbm(rng, ns, k_ii, k_ij)
    A_sparse = sparse.BCOO.from_scipy_sparse(A)
    A = jnp.asarray(A.todense())
    y = jax.nn.one_hot(jnp.asarray(g), num_classes=num_classes)
    
    # construct precision matrix
    Sig_node = sig_node**2 * (rho_node * jnp.ones((num_features, num_features)) + (1 - rho_node) * jnp.eye(num_features))
    Lam_node = jnp.linalg.inv(Sig_node)
    Lam_edge = beta * Lam_node
    Lam_gmrf = construct_precision(A, Lam_edge, Lam_node)
    
    # construct the mean vector
    mu_gmrf = jnp.concatenate([mu for n, mu in zip(ns, [mu0, mu1]) for _ in range(n)])
    
    # sample node features
    key, key_gmref = jax.random.split(key)
    X = sample_gmrf(key_gmref, mu_gmrf, Lam_gmrf, num_samples=num_samples)
    
    # determine input moments
    muX, SigX = estimate_moments(X)


    # create train/test split
    key, key_ = jax.random.split(key)
    train_idx_bool = jax.random.bernoulli(key_split, train_frac, shape=(num_nodes,)).astype(bool)
    train_idx = jnp.where(train_idx_bool)[0]
    test_idx =  jnp.where(~train_idx_bool)[0]

    # save dataset
    result_dict = {'data' : {
                    'muX' : muX,
                    'SigX' : SigX,
                    'y' : y,
                    'A' : A,
                    'Xsamples' : X,
                    'train_idx' : train_idx,
                    'test_idx' : test_idx
                     },
                'params' : {
                    'network_number' : i,
                    'total_networks' : num_networks,
                    'num_nodes' : num_nodes,
                    'ns' : ns,
                    'k_ii' : k_ii,
                    'k_ij' : k_ij,
                    'mu0' : mu0,
                    'mu1' : mu1,
                    'sig_node' : sig_node,
                    'rho_node' : rho_node,
                    'beta' : beta,
                    'num_features' : num_features,
                    'num_classes' : num_classes,
                    'num_samples' : num_samples,
                    'train_frac' : train_frac,
                    'seed_jax' : seed_jax,
                    'seed_numpy' : seed_numpy
                    }
                }
    
    
    with open(f'{data_dir}/sbm_corr={correlation}_graphnum={i}.pkl', 'wb') as f:
    
        pickle.dump(result_dict, f)