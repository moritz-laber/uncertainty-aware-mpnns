"""
sgc_lipschitzness.py
----------

Experiments & plots on the Lipschitz continuity of SGCs in FCD.
"""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse
import matplotlib.pyplot as plt
import numpy as np
import optax 
import ot
import scipy as sp
import pickle

from models import *
from utils import *

### PARAMETERS ###

plot_dir = './plots/sgc_lipschitzness/'      # directory to save plots
result_dir = './results/sgc_lipschitzness/'  # directory to save results

num_nodes = 256       # number of nodes
num_features = 3      # number of features
num_layers_sgc = 2    # number of SGC layers
num_classes = 2       # number of classes
num_samples = 1000    # number of samples

ns = [                # group sizes
    num_nodes//2,
    num_nodes - num_nodes//2
    ]
k_ii = 8              # average number of intra-community neighbors
k_ij = 4              # average number of inter-community neighbors

mu0 = -2.0 * jnp.ones(num_features)   # feature mean for class 0
mu1 =  2.0 * jnp.ones(num_features)   # feature mean for class 1

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

train_sample = 0    # index of sample to train on
train_frac = 0.7    # fraction of nodes in the training set
num_reps = 1000     # number of training iterations
eta = 1e-3          # learning rate

p = 2                     # p in the Wasserstein-p distance
max_samples_fcd = 100     # maximum number of samples to use in FCD and Wasserstein distance

figsize = (5,5)           # plot parameters
alpha = 0.25
fs = 12
loc = 'upper right'
colors = {
    'no-nodecorr_no-edgecorr':'#017a79',
    'nodecorr_no-edgecorr':'#94568c',
    'nodecorr_edgecorr':'#fec615'
    }

xplot_max = 0.15

seed_jax = 26216
seed_numpy = 1124

### MAIN ###

### Data Generation ###

rng = np.random.default_rng(seed_numpy)
key = jax.random.PRNGKey(seed=seed_jax)

# sample graph from SBM
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

## Train Models ##

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
Xdata = X[:, train_sample].reshape(num_nodes, num_features)
sgc, sgc_loss_list = train(sgc, optax.adabelief(eta), (A_sparse, Xdata, y), train_idx, num_reps=num_reps)

test_acc = accuracy(sgc, (A_sparse, Xdata, y), test_idx)
train_acc = accuracy(sgc, (A_sparse, Xdata, y), train_idx)
print(f'Train accuracy: {train_acc*100:.2f}%, Test accuracy: {test_acc*100:.2f}%')

# compute the logits for all samples
logits = jax.vmap(
    lambda x : sgc(
        x.reshape(num_nodes, num_features),
        A_sparse,
        final_embedding=False
        ).reshape(-1)
        )(X.T).T

## Compute Distances ##

# compute Feature Convolution Distance
dXS = FCD(X, A_sparse, max_samples=max_samples_fcd, p=p, node_subset=None)

# compute Wasserstein 2 distance on logits
max_samples = min(max_samples_fcd, num_samples)
logits = np.asarray(logits).reshape(num_nodes, num_classes, num_samples)

if p==1:
    metric = 'euclidean'
elif p==2:
    metric = 'sqeuclidean'
else:
    raise ValueError('p must be 1 or 2.')

dlogits = np.zeros((num_nodes, num_nodes))
for i in tqdm.tqdm(range(num_nodes)):
    for j in range(i+1, num_nodes):

        M = ot.dist(logits[i, :, :max_samples].T, logits[j, :, :max_samples].T, metric=metric)
        G = ot.emd(ot.unif(max_samples), ot.unif(max_samples), M)
        dlogits[i, j] = np.sqrt(np.sum(G * M)).item()

dlogits = np.asarray(dlogits)
dlogits = dlogits[np.triu_indices(num_nodes, k=1)]

## Compute Lipschitz Constant ##
C = np.max(dlogits/dXS)
print(f'Lipschitz constant: {C:.4f}')
print(f'Correlation Spearman: {sp.stats.spearmanr(dXS, dlogits)[0]:.4f}')
print(f'Correlation Pearson: {sp.stats.pearsonr(dXS, dlogits)[0]:.4f}')

## Save Results ##
with open(f'{result_dir}sgc_lipschitzness_{correlation}.pkl', 'wb') as f:
    pickle.dump({
        'dist_XS':dXS,
        'dist_logits':dlogits,
        'C_lipschitz':C,
        'spearman':sp.stats.spearmanr(dXS, dlogits)[0],
        'pearson':sp.stats.pearsonr(dXS, dlogits)[0]
    }, f)

## Plot ##
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(1,1,1)

ax.scatter(dXS, dlogits, alpha=alpha, color=colors[correlation])
ax.plot(np.linspace(0, xplot_max), C * np.linspace(0, xplot_max), color='k', ls='--', label=f'$C_L={C:.2f}$')



if p==1:
    ax.set_ylabel(r'$W_1(\xi_{i}^{(e)},\xi_{i}^{(e)})$', fontsize=fs)
    ax.set_xlabel(r'$\mathrm{FCD}_1(i,j)$', fontsize=fs)
elif p==2:
    ax.set_xlabel(r'$\mathrm{FCD}_2(i,j)$', fontsize=fs)
    ax.set_ylabel(r'$W_2(\xi_{i}^{(e)},\xi_{i}^{(e)})$', fontsize=fs)
else:
    ax.set_xlabel(r'$\mathrm{FCD}_p(i,j)$', fontsize=fs)
    ax.set_ylabel(r'$W_p(\xi_{i}^{(e)},\xi_{i}^{(e)})$', fontsize=fs)

ax.tick_params(axis='both', which='major', labelsize=fs-2)
ax.set_xlim(0., None)
ax.set_ylim(0., 1.7)

ax.legend(fontsize=fs-2, loc=loc)
fig.tight_layout()

fig.savefig(f'{plot_dir}sgc_lipschitzness_{correlation}.pdf')
fig.savefig(f'{plot_dir}sgc_lipschitzness_{correlation}.svg')  
fig.savefig(f'{plot_dir}sgc_lipschitzness_{correlation}.png', dpi=300)

