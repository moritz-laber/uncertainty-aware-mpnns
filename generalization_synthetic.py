"""
generalization_synthetic.py
----------

Conducts generalization experiments on synthetic data.
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
model_dir = './models/'                  # directory with models
data_dir = './data/synthetic/'           # directory with data
output_dir = './experiments/synthetic/'  # output directory

corr = sys.argv[1]                 # correlation type
num_layers = int(sys.argv[2])      # number of layers
num_reps = int(sys.argv[3])        # number of training repetitions
num_graphs = 6                     # number of graphs to consider, including the training graph.
train_graph_num = 0                # index of the training graph
num_samples_fcd = 100              # number of samples for estimating Wasserstein distance
Cl = 1.0                           # lipschitz constant of the loss
p = 1                              # which metric to use for the Wasserstein distance
epsilon = 0.99                     # uniform robustness constant 
delta = 0.05                       # error probability
seed = 125124                      # jax seed : dummy

### MAIN SCRIPT ###

## seeding ##
key = jax.random.PRNGKey(seed)
key, key_model = jax.random.split(key)

## Checks ##
assert train_graph_num==0, "Please reorder graph s.t. the training graph is the first."

## Load hyperparameters ##
with open(f'{model_dir}{corr}/hyperparams_model=sgc_corr={corr}_l={num_layers}_reps={num_reps}.pkl', 'rb') as f:
    hyperparams = pickle.load(f)

num_features = hyperparams['params']['num_features']
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
    f'{model_dir}{corr}/weights_model=sgc_corr={corr}_l={num_layers}_reps={num_reps}.eqx',
    sgc
    )


S = 0          # sample size
Chi = 0        # chromatic number
dX = []        # distances in FCD
dlosses = []   # distances in loss space
dlogits = []   # distances in logit space
lhs = []       # empirical estimate of the LHS

for graph_num in range(num_graphs):

    ## Load Data ##
    with open(f'{data_dir}sbm_corr={corr}_graphnum={graph_num}.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    Adense = dataset['data']['A']
    A_sparse = sparse.BCOO.fromdense(Adense)
    X = dataset['data']['muX'].reshape(num_nodes, num_features)
    y = dataset['data']['y']
    Xsamples = dataset['data']['Xsamples']
    test_idx = dataset['data']['test_idx']
    train_idx = dataset['data']['train_idx']
    num_test_nodes = test_idx.shape[0]
    num_train_nodes = train_idx.shape[0]

    # update sample size on the train graph
    if graph_num == train_graph_num:
        S = num_train_nodes
        Chi = num_train_nodes

    # check number of available samples
    if num_samples_fcd > Xsamples.shape[1]:
        print(f"Only {Xsamples.shape[1]} < {num_samples_fcd} samples available.")
        num_samples_fcd = Xsamples.shape[1]
    else:
        Xsamples = Xsamples[:, :num_samples_fcd]

    # compute feature convolution distance (input space)
    dX_g = FCD(
        Xsamples,
        A_sparse,
        max_samples=num_samples_fcd,
        p=p,
        node_subset=np.arange(num_nodes)
    )
    dX.append(dX_g)

    # get losses over all samples
    logits = jax.vmap(lambda X: sgc(X.reshape(num_nodes, num_features), A_sparse, final_embedding=True), in_axes=1, out_axes=2)(Xsamples)
    yprobs = jax.nn.softmax(logits, axis=1)
    losses = -jnp.sum(y[:,:,None] * jnp.log(yprobs), axis=1)

    # compute the Wasserstein distance in loss space between all nodes
    dlosses_g = wasserstein_sample(
        losses,
        losses,
        p=p
    )
    dlosses.append(dlosses_g)

    # compute the Wasserstein distance in logit space between all nodes
    dlogits_g = wasserstein_sample(
        logits,
        logits,
        p=p
    )
    dlogits.append(dlogits_g)

    if graph_num == train_graph_num:
        # extract the loss on the training set
        train_idx = dataset['data']['train_idx']
        train_losses = losses[train_idx, :]
    else:
        # compute the Wasserstein distance between the 
        # test and train loss distributions to estimate
        # the LHS empirically.
        lhs_g = wasserstein_sample(
            losses,
            train_losses,
            p=p
        )
        
        lhs.append(lhs_g)


lhs = jnp.concatenate(lhs)
dX = jnp.concatenate(dX)
dlosses = jnp.concatenate(dlosses)
dlogits = jnp.concatenate(dlogits)

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

## Save Results ##
with open(f'{output_dir}{corr}/generalization_model=sgc_corr={corr}_l={num_layers}_reps={num_reps}.pkl', 'wb') as f:
    pickle.dump({
        'corr' : corr,
        'num_classes' : num_classes,
        'num_features' : num_features,
        'dist_XS':dX,
        'dist_logits':dlosses,
        'lipschitz_FCD': CL,
        'loss_lipschitz_l2' : Cl,
        'M' : M, 
        'epsilon' : epsilon,
        'delta' : delta,
        'Chi' : Chi,
        'K' : K,
        'bound' : bound,
        'empirical_LHS' : lhs,
        'spearman':sp.stats.spearmanr(dX, dlosses)[0],
        'pearson':sp.stats.pearsonr(dX, dlosses)[0]
    }, f)




    