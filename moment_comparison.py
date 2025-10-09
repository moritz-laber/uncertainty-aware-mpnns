"""
moment_comparison.py
----------

Experiments on moment propagation methods for MPNNs.
"""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse
import numpy as np
import optax
import pickle

from models import *
from utils import *

### PARAMETERS ###

result_dir = './results/moment_comparison/'    # directory to save results

num_nodes = 256         # number of nodes
num_features = 3        # number of node features
num_layers_gcn = 2      # number of GCN layers
num_hidden = 16         # number of hidden units
num_classes = 2         # number of classes
num_samples = 10000     # number of samples for moment estimation


ns = [                  # community sizes
    num_nodes//2,
    num_nodes - num_nodes//2
    ]
k_ii = 8                # average number of intra community neighbors
k_ij = 4                # average number of inter community neighbors

nonlinearities = [      # nonlinearities to consider
    jax.nn.relu,
    jax.nn.tanh,
    jax.nn.gelu
    ]

mu0 = -2.0 * jnp.ones(num_features)       # mean for class 0
mu1 =  2.0 * jnp.ones(num_features)       # mean for class 1

correlation = 'ndnf'       # correlation type: inif, indf, dndf

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

eps = 1e-20                # truncation of eigenvalues for numerical stability of Wasserstein distance
beta_relu = 20.0           # parameter for softplus
eps_taylor = 1e-6          # parameter for numerical stability in Taylor

train_sample = 0    # index of the sample to train on
train_frac = 0.7    # fraction of nodes in the training set
num_reps = 1000     # number of training iterations
eta = 1e-3          # initial learning rate of AdaBelief

seed_jax = 26216    # jax seed
seed_numpy = 1124   # numpy seed (for SBM)

### MAIN ###

## Generate Data ##

# seeding
rng = np.random.default_rng(seed_numpy)
key = jax.random.PRNGKey(seed=seed_jax)

# sample graph from SBM
A, g = sbm(rng, ns, k_ii, k_ij)
A_sparse = sparse.BCOO.from_scipy_sparse(A)
A = jnp.asarray(A.todense())
y = jax.nn.one_hot(jnp.asarray(g), num_classes=2)

# construct precision matrix
Sig_node = sig_node**2 * (rho_node * jnp.ones((num_features, num_features)) + (1 - rho_node) * jnp.eye(num_features))
Lam_node = jnp.linalg.inv(Sig_node)
Lam_edge = beta * Lam_node
Lam_gmrf = construct_precision(A, Lam_edge, Lam_node)

# construct the mean vector
mu_gmrf = jnp.concatenate([mu for n, mu in zip(ns, [mu0, mu1]) for _ in range(n)])

# sample node features
key, key_gmref = jax.random.split(key)

for nonlinearity in nonlinearities:

    X = sample_gmrf(key_gmref, mu_gmrf, Lam_gmrf, num_samples=num_samples)
    muX, SigX = estimate_moments(X)

    if nonlinearity == jax.nn.gelu:
        nonlin_str = 'gelu'
    if nonlinearity == jax.nn.relu:
        nonlin_str = 'relu'
    if nonlinearity == jax.nn.tanh:
        nonlin_str = 'tanh'

    ## Train Model ##
    print('training ', nonlin_str)

    # create the train/test split on a particular sample
    key, key_ = jax.random.split(key)
    train_idx_bool = jax.random.bernoulli(key_, train_frac, shape=(num_nodes,)).astype(bool)
    train_idx = jnp.where(train_idx_bool)[0]
    test_idx =  jnp.where(~train_idx_bool)[0]

    # initialize model
    key, key_ = jax.random.split(key)
    gcn = GCN(
        key_,
        dims=[num_features] + num_layers_gcn* [num_hidden] + [num_classes],
        non_linearity=nonlinearity,
        final_non_linearity=jax.nn.softmax,
    )

    # train model
    Xdata = X[:, train_sample].reshape(num_nodes, num_features)
    gcn, gcn_loss_list = train(gcn, optax.adabelief(eta), (A_sparse, Xdata, y), train_idx, num_reps=num_reps)
    test_acc = accuracy(gcn, (A_sparse, Xdata, y), test_idx)
    train_acc = accuracy(gcn, (A_sparse, Xdata, y), train_idx)

    ## Moment Propagation ##

    # define flattened model
    f = flatten_model(
        gcn,
        A_sparse
    )

    f_approx = flatten_model(
        gcn,
        A_sparse,
        alternative_nonlinearity=(lambda x: softplus(x, beta=beta_relu)) if nonlinearity==jax.nn.relu else nonlinearity
    )

    # sampling
    print('sampling')
    muY_sample, SigY_sample = sample_propagation(
        f,
        X
        )

    # Multi-dimensional Tailor
    print('multidimensional Taylor')
    muY_lin, SigY_lin = linear_propagation(
        f_approx,
        muX,
        SigX,
        eps=eps
        )
    
    muY_quad, SigY_quad = quadratic_propagation(
        f_approx,
        muX,
        SigX,
        eps=eps
        )
    
    
    # PTPE
    print('PTPE')
    muY_ptpe, SigY_ptpe = ptpe_gcn(
        gcn,
        A_sparse,
        True,
        muX,
        SigX
        )
    
    # Layerwise Taylor
    print('layerwise Taylor')
    muY_tay1, SigY_tay1 = taylor_gcn(
        gcn,
        A_sparse,
        True,
        muX,
        SigX,
        order=1,
        alternative_nonlinearity=(lambda x: softplus(x, beta=beta_relu)) if nonlinearity==jax.nn.relu else None
        )
    
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
    

    ### Analysis ###
    print('Analysis')

    # mean
    relative_errors_mu = [
        relative_error(muY_lin, muY_sample),
        relative_error(muY_quad, muY_sample),
        relative_error(muY_ptpe, muY_sample),
        relative_error(muY_tay1, muY_sample),
        relative_error(muY_tay2, muY_sample),
        relative_error(muY_tay2_gc, muY_sample)
        ]

    fro_errors_mu = [
        jnp.linalg.norm(muY_lin - muY_sample),
        jnp.linalg.norm(muY_quad - muY_sample),
        jnp.linalg.norm(muY_ptpe - muY_sample),
        jnp.linalg.norm(muY_tay1 - muY_sample),
        jnp.linalg.norm(muY_tay2 - muY_sample),
        jnp.linalg.norm(muY_tay2_gc - muY_sample)
        ]
    
    # Covariance
    relative_errors_sig = [
        relative_error(SigY_lin, SigY_sample),
        relative_error(SigY_quad, SigY_sample),
        relative_error(SigY_ptpe, SigY_sample),
        relative_error(SigY_tay1, SigY_sample),
        relative_error(SigY_tay2, SigY_sample),
        relative_error(SigY_tay2_gc, SigY_sample)
        ]

    fro_errors_sig = [
            jnp.linalg.norm(SigY_lin - SigY_sample),
            jnp.linalg.norm(SigY_quad - SigY_sample),
            jnp.linalg.norm(SigY_ptpe - SigY_sample),
            jnp.linalg.norm(SigY_tay1 - SigY_sample),
            jnp.linalg.norm(SigY_tay2 - SigY_sample),
            jnp.linalg.norm(SigY_tay2_gc - SigY_sample)
            ]
    
    # Wasserstein 2
    wasserstein_dist_result = [  
            wasserstein_distance((muY_lin, SigY_lin), (muY_sample, SigY_sample), eps=eps),
            wasserstein_distance((muY_quad, SigY_quad), (muY_sample, SigY_sample), eps=eps),
            wasserstein_distance((muY_ptpe, SigY_ptpe), (muY_sample, SigY_sample), eps=eps),
            wasserstein_distance((muY_tay1, SigY_tay1), (muY_sample, SigY_sample), eps=eps),
            wasserstein_distance((muY_tay2, SigY_tay2), (muY_sample, SigY_sample), eps=eps),
            wasserstein_distance((muY_tay2_gc, SigY_tay2_gc), (muY_sample, SigY_sample), eps=eps)
            ]

    ## Save Results ###

    result_dict = {'results': {
                    'rel_mu' : relative_errors_mu,
                    'fro_mu' : fro_errors_mu,
                    'rel_sig' : relative_errors_sig,
                    'fro_sig' : fro_errors_sig,
                    'wasserstein' : wasserstein_dist_result,
                    'order' :  ['all_lin', 'all_quad_trunc', 'ptpe', 'layer_lin', 'layer_quad_trunc', 'layer_quad_gc']
                     },
                'params' : {
                    'num_nodes' : num_nodes,
                    'k_ii' : k_ii,
                    'k_ij' : k_ij,
                    'sig_node' : sig_node,
                    'rho_node' : rho_node,
                    'beta' : beta,
                    'beta_relu' : beta_relu if nonlinearity==jax.nn.relu else None,
                    'nonlinearity' : nonlin_str,
                    'num_features' : num_features,
                    'num_hidden' : num_hidden,
                    'num_layers' : num_layers_gcn,
                    'num_classes' : num_classes,
                    'num_samples' : num_samples,
                    'train_frac' : train_frac,
                    'test_acc' : test_acc,
                    'train_acc' : train_acc
                }
                }


    with open(f'{result_dir}comparison_corr={correlation}_fun={nonlin_str}.pkl', 'wb') as f:

        pickle.dump(result_dict, f)



