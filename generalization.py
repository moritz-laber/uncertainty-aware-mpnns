"""
generalization.py
----------

Experiments on generalization of SGC in the uncertainty aware setting.

When using this code please cite:
Chernikova et al. (2025) Robustness and Generalization in Uncertainty-Aware Message Passing Neural Networks.

author: Moritz Laber
date: October 2025
"""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse
import matplotlib.pyplot as plt
import numpy as np
import optax
import ot
import pickle
import scipy as sp

from models import *
from utils import *

### PARAMETERS ###

plot_dir = './plots/sgc_L=3_generalization/'        # plot directory
result_dir = './results/sgc_L=3_generalization/'    # result directory

num_graphs = 6              # number of graphs 1 train n-1 test
num_nodes = 256             # number of nodes
num_features = 3            # number of features
num_layers_sgc = 3          # number of layers in SGC
num_classes = 2             # number of classes
num_samples = 100           # number of samples

ns = [                      # number of nodes in each group
    num_nodes//2,
    num_nodes - num_nodes//2
    ]
k_ii = 8                   # expected number of intra group neighbors
k_ij = 4                   # expected number of inter group neighbors

mu0 = -2.0 * jnp.ones(num_features)    # mean of features in group 0
mu1 =  2.0 * jnp.ones(num_features)    # mean of features in group 1

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

train_sample = 0     # the sample to train on
train_frac = 0.5     # the fraction of nodes to train on
num_reps = 1000      # number of training repetitions
eta = 1e-3           # learning rate

p = 1                           # constants in the generalization bound theorem
epsilon = 0.99                  #
delta = 0.1                     #
fix_Chi = None                  # the largest number of sampled nodes will be used
max_samples_fcd = num_samples   # number of samples for distance estimation

figsize = (5,5)                 # plot parameters
alpha = 0.25
fs = 12
loc = 'upper right'
colors = {
    'no-nodecorr_no-edgecorr':'#017a79',
    'nodecorr_no-edgecorr':'#94568c',
    'nodecorr_edgecorr':'#fec615'
    }
xplot_max = 0.15

seed_jax = 26216    # seed jax
seed_numpy = 1124   # seed numpy (graph generation)



### MAIN ###

## Compute Covering Number ##
f0 = float(num_features)
theta = f0 * (5.0 + np.log(f0) + np.log(np.log(f0)))
C1 = np.power(3, f0) * np.exp(1.) * theta
C2 = (f0 + 1.) * np.log(3.0) + (p + 2.0) * np.log(2.0) + np.log(theta)

logK = (
        C1  * np.power(epsilon, -f0)
            * np.power(np.log(1.0/epsilon), f0) 
            * ((f0 + 1.0) * np.log((1.0/epsilon)*np.log(1.0/epsilon)) + C2)
     )

K = np.exp(logK)

## Generate Data ##

rng = np.random.default_rng(seed_numpy)
key = jax.random.PRNGKey(seed=seed_jax)

# sample graph from SBM
As = []
As_sparse = []
ys = []
Xs = []
muXs, SigXs = [], []
for i in range(num_graphs):

    A, g = sbm(rng, ns, k_ii, k_ij)
    A_sparse = sparse.BCOO.from_scipy_sparse(A)
    A = jnp.asarray(A.todense())
    y = jax.nn.one_hot(jnp.asarray(g), num_classes=num_classes)

    # construct precision matrix
    Sig_node = sig_node**2 * (rho_node * jnp.ones((num_features, num_features)) + (1 - rho_node) * jnp.eye(num_features))
    Lam_node = jnp.linalg.inv(Sig_node)
    Lam_edge = beta * Lam_node
    Lam_gmrf= construct_precision(A, Lam_edge, Lam_node)

    # construct the mean vector
    mu_gmrf = jnp.concatenate([mu for n, mu in zip(ns, [mu0, mu1]) for _ in range(n)])

    # sample node features
    key, key_ = jax.random.split(key)
    X = sample_gmrf(key_, mu_gmrf, Lam_gmrf, num_samples=num_samples)

    # estimate mean and covariance from samples
    muX, SigX = estimate_moments(X)

    # store
    As_sparse.append(A_sparse)
    ys.append(y)
    Xs.append(X)
    muXs.append(muX)
    SigXs.append(SigX)

## Train Model ##

# create the train/test split on a particular sample
key, key_ = jax.random.split(key)
train_idx_bool = jax.random.bernoulli(key_, train_frac, shape=(num_nodes,)).astype(bool)
train_idx = jnp.where(train_idx_bool)[0]
test_idx =  jnp.where(~train_idx_bool)[0]

# initialize model
key, key_ = jax.random.split(key)
sgc = SGC(
    key_,
    n_layers=num_layers_sgc,
    n_features=num_features,
    n_classes=num_classes,
    non_linearity=jax.nn.softmax
)

# train model
Xdata = Xs[0][:, train_sample].reshape(num_nodes, num_features)
sgc, sgc_loss_list = train(sgc, optax.adabelief(eta), (As_sparse[0], Xdata, y), train_idx, num_reps=num_reps)

test_acc = accuracy(sgc, (As_sparse[0], Xdata, y), test_idx)
train_acc = accuracy(sgc, (As_sparse[0], Xdata, y), train_idx)

# compute the loss on all samples
dlosses = []
dXS = []
S = 0
Chi = 0
for i in range(1, num_graphs):

    # sample the test set
    key, key_ = jax.random.split(key)
    train_idx_bool = jax.random.bernoulli(key_, train_frac, shape=(num_nodes,)).astype(bool)
    train_idx = jnp.where(train_idx_bool)[0]
    test_idx =  jnp.where(~train_idx_bool)[0]
    S += test_idx.shape[0]
    Chi = max(Chi, test_idx.shape[0])

    logits = jax.vmap(
        lambda x : sgc(
            x.reshape(num_nodes, num_features),
            As_sparse[i],
            final_embedding=False
            ).reshape(-1)
            )(Xs[i].T).T

    log_probs = jax.vmap(lambda x: jax.nn.log_softmax(x, axis=1))(logits.reshape(num_nodes, num_classes, num_samples).T).T
    losses = jax.vmap(lambda x : -jnp.sum(x.T * ys[i], axis=1))(log_probs.T).T

    ## Compute Distances ##

    # compute Feature Convolution Distance
    dXS_g = FCD(Xs[i], As_sparse[i], max_samples=max_samples_fcd, p=p, node_subset=np.asarray(test_idx))
    dXS.append(dXS_g)

    # compute Wasserstein 2 distance on logits
    max_samples = min(max_samples_fcd, num_samples)
    logits = np.asarray(logits).reshape(num_nodes, num_classes, num_samples)

    if p==1:
        metric = 'euclidean'
    elif p==2:
        metric = 'sqeuclidean'
    else:
        raise ValueError('p must be 1 or 2.')

    dlosses_g = []
    for i in tqdm.tqdm(np.asarray(test_idx)):
        for j in np.asarray(test_idx):
            if i<j:
                M = ot.dist(np.asarray(losses[i, None, :max_samples].T), np.asarray(losses[j, None, :max_samples].T), metric=metric)
                G = ot.emd(ot.unif(max_samples), ot.unif(max_samples), M)
                dlosses_g.append(np.sqrt(np.sum(G * M)).item())

    dlosses_g = np.asarray(dlosses_g)
    dlosses.append(dlosses_g)

dlosses = jnp.concatenate(dlosses)
dXS = jnp.concatenate(dXS)
M = np.max(dlosses)

## Estimate the Lipschitz constant ##
C = np.max(dlosses/dXS)
print(f'Lipschitz constant: {C:.4f}')
print(f'Correlation Spearman: {sp.stats.spearmanr(dXS, dlosses)[0]:.4f}')
print(f'Correlation Pearson: {sp.stats.pearsonr(dXS, dlosses)[0]:.4f}')

## Compute the theoretical generalization bound ##

if fix_Chi is not None:
    Chi = fix_Chi

bound = (
    2 * C * epsilon
    + M * jnp.sqrt((Chi/S) * (2.0 * (K+1) * jnp.log(2.0) + 2.0 * jnp.log(1.0/delta)))
    )

print(f'Bound: {bound:.4f}')

## Save Results ##
with open(f'{result_dir}sgc_generalization_{correlation}.pkl', 'wb') as f:
    pickle.dump({
        'dist_XS':dXS,
        'dist_logits':dlosses,
        'C_lipschitz': C,
        'M' : M, 
        'bound' : bound,
        'epsilon' : epsilon,
        'delta' : delta,
        'K' : K,
        'sigma_node' : sig_node,
        'rho_node' : rho_node,
        'beta' : beta,
        'spearman':sp.stats.spearmanr(dXS, dlosses)[0],
        'pearson':sp.stats.pearsonr(dXS, dlosses)[0]
    }, f)
