"""
generalization_realworld.py
----------

Conducts generalization experiments on real-world data.
"""

### IMPORTS ###
import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse
import numpy as np
import optax
import pickle
import sys

from models import *
from utils import *

### PARAMETERS ###
model_dir = './models/'                 # directory with models
data_dir = './data/'                    # directory with data
output_dir = './experiments/realworld/' # directory for output

dataset = sys.argv[1]            # dataset name
num_layers = int(sys.argv[2])    # number of layers
num_reps = int(sys.argv[3])      # number of training repetitions
noise_level = 0.05               # noise level
num_samples_fcd = 100            # number of samples to estimate Wasserstein distance
Cl = 1.0                         # loss function Lipschitz constant
p = 1                            # metric used in the Wasserstein distance
epsilon = 0.99                   # uniform robustness constant
delta = 0.05                     # error probability
seed = 26216                     # jax seed : 26216 for consistency

### MAIN SCRIPT ###

## seeding ##
# splits are done this way to get the same train test split
# as in the training script.
key = jax.random.PRNGKey(seed)
key, key_model, key_split = jax.random.split(key, 3)
key, key_samples = jax.random.split(key)

## Load hyperparameters ##
with open(f'{model_dir}{dataset}/hyperparams_model=sgc_data={dataset}_l={num_layers}_reps={num_reps}.pkl', 'rb') as f:
    hyperparams = pickle.load(f)

num_features = hyperparams['params']['num_features']
which_split = hyperparams['params']['which_split']
train_frac = hyperparams['params']['train_frac']
num_nodes = hyperparams['params']['num_nodes']
num_classes = hyperparams['params']['num_classes']

## Load model ##
sgc = SGC(
    key_model,
    n_layers=num_layers,
    n_features=num_features,
    n_classes=num_classes,
    non_linearity=jax.nn.softmax
)

sgc = eqx.tree_deserialise_leaves(
    f'{model_dir}{dataset}/weights_model=sgc_data={dataset}_l={num_layers}_reps={num_reps}.eqx',
    sgc
    )

## Load Data ##
data = load_dataset(
    f'{data_dir}{dataset}.npz',
    num_features=num_features,
    which_split=which_split,
    train_frac=train_frac,
    key_split=key_split
    )
A_sparse, X, y, train_idx, test_idx = data
num_nodes = A_sparse.shape[0]
num_classes = y.shape[1]
X = X.reshape(num_nodes, num_features)
num_test_nodes = test_idx.shape[0]
num_train_nodes = train_idx.shape[0]

# update sample size
S = num_train_nodes

# update chromatic number
Chi = num_train_nodes

# sample iid noise on top of the features
noise = noise_level * jnp.abs(X)[:,:,None] * jax.random.normal(key_samples, shape=(X.shape[0], X.shape[1], num_samples_fcd))
Xsamples = X[:,:,None] + noise

# compute feature convolution distance (input space)
dX = FCD(
    Xsamples.reshape(num_nodes * num_features, num_samples_fcd),
    A_sparse,
    max_samples=num_samples_fcd,
    p=p,
    node_subset=np.arange(num_nodes)
)

# get losses over all samples
logits = jax.vmap(lambda X: sgc(X.reshape(num_nodes, num_features), A_sparse, final_embedding=True), in_axes=2, out_axes=2)(Xsamples)
yprobs = jax.nn.softmax(logits, axis=1)
losses = -jnp.sum(y[:,:,None] * jnp.log(yprobs), axis=1)

# compute the Wasserstein distance in loss space between all nodes
dlosses = wasserstein_sample(
    losses,
    losses,
    p=p
)

# compute the Wasserstein distance in logit space between all nodes
dlogits = wasserstein_sample(
    logits,
    logits,
    p=p
)

# extract the loss on the training set
train_losses = losses[train_idx, :]
test_losses = losses[test_idx, :]

lhs = wasserstein_sample(
    test_losses,
    train_losses,
    p=p
)

# Estimate the Lipschitz Constant in FCD Distance
CL = jnp.max(dlogits[dX>0]/dX[dX>0])

# Compute the largest loss distance. This is a rough estimate of the actually required
# intractable value of M in Theorem 6.
M = jnp.max(dlosses)

## Compute the theoretical generalization bound ##

# compute the covering number
f0 = float(num_features)
theta = f0 * (5.0 + np.log(f0) + np.log(np.log(f0)))
C1 = np.power(3, f0) * np.exp(1.) * theta
C2 = (f0 + 1.) * np.log(3.0) + (p + 2.0) * np.log(2.0) + np.log(theta)

logK = (
        C1  * np.power(epsilon, -f0)
            * np.power(np.log(1.0/epsilon), f0) 
            * ((f0 + 1.0) * np.log((1.0/epsilon)*np.log(1.0/epsilon)) + C2)
     )

K = num_classes * np.exp(logK)

# compute the bound
bound = (
    2 * CL * Cl * epsilon
    + M * jnp.sqrt((Chi/S) * (2.0 * (K+1) * jnp.log(2.0) + 2.0 * jnp.log(1.0/delta)))
    )

print(f'Bound: {bound:.4f}')

## Save Results ##
with open(f'{output_dir}{dataset}/generalization_model=sgc_data={dataset}_l={num_layers}_reps={num_reps}.pkl', 'wb') as f:
    pickle.dump({
        'dataset' : dataset,
        'noise_level' : noise_level,
        'num_layers' : num_layers,
        'num_classes' : num_classes,
        'num_features' : num_features,
        'dist_XS': dX,
        'dist_logits': dlosses,
        'lipschitz_FCD': CL,
        'loss_lipschitz_l2' : Cl,
        'M' : M, 
        'epsilon' : epsilon,
        'delta' : delta,
        'K' : K,
        'Chi' : Chi,
        'bound' : bound,
        'empirical_LHS' : lhs,
        'spearman':sp.stats.spearmanr(dX, dlosses)[0],
        'pearson':sp.stats.pearsonr(dX, dlosses)[0],
        'seed' : seed
    }, f)