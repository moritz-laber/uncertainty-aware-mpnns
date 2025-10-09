"""
robustness_radius.py
----------

Experiments & plots on the robustness radius of uncertainty aware GCNs.
"""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse
import matplotlib.pyplot as plt
import numpy as np
import optax
import pickle

from models import *
from utils import *

### PARAMETERS ###

result_dir = './results/robustness/'        # directory to save results
plot_dir = './plots/robustness/'            # directory to save plots

num_nodes = 256             # number of nodes
num_features = 3            # number of node features 
num_layers_gcn = 2          # number of GCN layers
num_hidden = 16             # number of hidden units
num_classes = 2             # number of output classes
num_samples = 10000         # number of samples (used for moment estimation)

ns = [                      # community sizes
    num_nodes//2,
    num_nodes - num_nodes//2
    ]

k_ii = 8                    # average number of intra community neighbors
k_ij = 4                    # average number of inter community neighbors

nonlinearities = [          # nonlinearities to consider. Only tanh, gelu, relu are supported
    jax.nn.tanh,
    jax.nn.relu,
    jax.nn.gelu
    ]

mu0 = -2.0 * jnp.ones(num_features) # mean for class 1
mu1 =  2.0 * jnp.ones(num_features) # mean for class 2


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

delta = 0.05       # 1-delta is the misclassification probability
beta_relu = 20     # parameter in the approximation of relu for Taylor expansion

train_sample = 0    # on which sample to train the model
train_frac = 0.7    # fraction of nodes in the training set
num_reps = 1000     # number of training iterations
eta = 1e-3          # initial learning rate of AdaBelief

nbins = 30                  # number of bins for histogram
fs = 12                     # fontsize for plots
text_pos = (0.02, 0.85)     # position of the text in the plot
arrow_pos_1 = (0.01, 0.95)  # position of the arrow in the plot
arrow_pos_2 = (0.15, 0.95)  # position of the arrow in the plot
fs_text = 8                 # fontsize of the text in the plot
ymax_plt = 0.175            # max y-value in the plot
xmin_plt = None             # min x-value in the plot

seed_jax = 26216    # jax seed
seed_numpy = 1124   # numpy seed (network generation)

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
key, key_ = jax.random.split(key)
X = sample_gmrf(key_, mu_gmrf, Lam_gmrf, num_samples=num_samples)

# estimate mean and covariance from samples
muX, SigX = estimate_moments(X)

for nonlinearity in nonlinearities:

    if nonlinearity == jax.nn.relu:
        color = '#0485d1'
        nonlin_str = 'relu'
    if nonlinearity == jax.nn.tanh:
        color = '#ff474c'
        nonlin_str = 'tanh'
    if nonlinearity == jax.nn.gelu:
        color = '#40a368'
        nonlin_str = 'gelu'
    
    print('Nonlinearity: ', nonlin_str)

    ## Train models ###

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


    # train models
    Xdata = X[:, train_sample].reshape(num_nodes, num_features)
    gcn, gcn_loss_list = train(gcn, optax.adabelief(eta), (A_sparse, Xdata, y), train_idx, num_reps=num_reps)
    test_acc = accuracy(gcn, (A_sparse, Xdata, y), test_idx)
    train_acc = accuracy(gcn, (A_sparse, Xdata, y), train_idx)

    ## Moment Propagation ##
    f = flatten_model(gcn, A_sparse)

    # sampling
    muY_sample, SigY_sample = sample_propagation(f, X)

    # ptpe
    muY_ptpe, SigY_ptpe = ptpe_gcn(
                        gcn,
                        A_sparse,
                        True,
                        muX,
                        SigX
                        )

    # second order Taylor 
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

    ## Lipschitzness ##

    # input distance
    dists = []
    for i in tqdm.tqdm(range(num_samples)):
        Xs = X[:, i].reshape(num_nodes, num_features)
        d = jnp.linalg.norm(Xs[:, None, i] - Xs[None, :, i], axis=-1)
        dists.append(d)

    dists_x = jnp.concat(dists, axis=0)

    # output distance
    f = lambda X: gcn(X.reshape(num_nodes, num_features), A_sparse, final_embedding=True).reshape(-1)
    Y = jax.vmap(f)(X.T).T

    dists = []
    for i in tqdm.tqdm(range(num_samples)):
        Ys = Y[:, i].reshape(num_nodes, num_classes)
        d = jnp.linalg.norm(Ys[:, None, i] - Ys[None, :, i], axis=-1)
        dists.append(d)

    dists_y = jnp.concat(dists, axis=0)

    # Lipschitz constant
    C = jnp.max(dists_y/dists_x)
    
    ## Compute robustness radius  ##
    for mu, Sig, estimator in zip([muY_ptpe, muY_tay2_gc, muY_sample],
                                [SigY_ptpe, SigY_tay2_gc, SigY_sample],
                                ['PTPE', 'tay2gc', 'sample']):
        
        print(f'Computing robustness for {estimator} estimator and {nonlin_str} nonlinearity.')

        # radius computation 
        mu = mu.reshape(num_nodes, num_classes)
        Sig = Sig.reshape(num_nodes, num_classes, num_nodes, num_classes)

        radii = []
        for v in range(num_nodes):
            ystar = jnp.where(y[v, :])[0][0].item()
            mu_hat = mu[v, ystar] - mu[v, 1-ystar]
            if mu_hat < 0:
                continue
            sig_hat = Sig[v, ystar, v, ystar] + Sig[v, 1-ystar, v, 1-ystar] - 2*Sig[v, ystar, v, 1-ystar]

            radii.append(jnp.max(jnp.asarray([(mu_hat - jnp.sqrt(sig_hat)*jnp.sqrt((1.0 - delta)/delta))/jnp.sqrt(2)*C])))

        radii = jnp.asarray(radii)

        ## Save results ##
        with open(f'{result_dir}{correlation}/robustness_{estimator}_{nonlin_str}.pkl', 'wb') as f:

            pickle.dump({
                'nonlinearity' : nonlin_str,
                'estimator' : estimator,
                'dist_y' : dists_y,
                'dist_x' : dists_x,
                'C_lipschitz' : C,
                'radii' : radii,
                'beta_relu' : beta_relu if nonlinearity==jax.nn.relu else None,
                'test_acc' : test_acc,
                'train_acc' : train_acc

            },
            f)

        ## Plot ##
        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(111)

        counts, bins = jnp.histogram(radii, bins=nbins, density=True)

        ax1.bar(0.5*(bins[1:] + bins[:-1]), counts, width=(bins[1]-bins[0]), edgecolor='darkslategray', color=color)
        
        ax1.set_xlabel(r'perturbation size $\epsilon$', fontsize=fs)
        ax1.set_ylabel(r'relative frequency $n(\epsilon)/n$', fontsize=fs)
        ax1.tick_params(axis='both', which='major', labelsize=fs-2)

        ax1.vlines(x=0, ymin=0, ymax=1.0, color='k', ls='--')
        ax1.annotate('', xy=arrow_pos_1, xycoords='axes fraction', xytext=arrow_pos_2, textcoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=1.5))
        ax1.text(*text_pos, transform=ax1.transAxes, s=f'no robustness\nguarantee', fontsize=fs_text)

        ax1.set_ylim(0.00, ymax_plt)
        ax1.set_xlim(xmin_plt, None)

        fig.tight_layout()

        fig.savefig(f'{plot_dir}{correlation}/robustnes_{estimator}_{nonlin_str}_{correlation}.svg')
        fig.savefig(f'{plot_dir}{correlation}/robustnes_{estimator}_{nonlin_str}_{correlation}.pdf')
        fig.savefig(f'{plot_dir}{correlation}/robustnes_{estimator}_{nonlin_str}_{correlation}.png', dpi=600)

