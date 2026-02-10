"""
train_sgc_synthetic.py
----------

Trains SGC model on synthetic datasets using
a given number of layers and iterations.
"""

### IMPORTS ###
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse
import numpy as np
import optax
import pickle

from models import *
from utils import *

import sys

### PARAMETERS ###
save_result = True                    # whether to save model and hyperparameters
inductive = True                      # whether to use an inductive split
data_dir = './data/synthetic/'        # directory with datasets
result_dir = './models/'              # directory to save trained model
graph_num = 0                         # network index
correlation = sys.argv[1]             # synthetic dataset identified by correlation
num_layers = 3                        # number of sgc layers
num_hidden = 16                       # number of hidden units
num_reps = 1000                       # number of training iterations
eta = 1e-3                            # initial learning rate of AdaBelief
seed = 26216                          # jax seed : 26216
test_graphs = [1,2,3,4,5]             # graphs to evaluate on in the inductive setting

### MAIN SCRIPT ###

## Seeding ##
key = jax.random.PRNGKey(seed=seed)

## Load Data ##
with open(f'{data_dir}sbm_corr={correlation}_graphnum={graph_num}.pkl', 'rb') as f:
    dataset = pickle.load(f)

Adense = dataset['data']['A']
A_train = sparse.BCOO.fromdense(Adense)
num_classes = dataset['params']['num_classes']
num_nodes = dataset['params']['num_nodes']
num_features = dataset['params']['num_features']

X = dataset['data']['muX']
y = dataset['data']['y']

if inductive:
    train_idx = jnp.arange(num_nodes)
else:
    train_idx = dataset['data']['train_idx']
    test_idx = dataset['data']['test_idx']

## Train Model ##

# initialize model
key, key_ = jax.random.split(key)
sgc = SGC(
    key_,
    n_layers=num_layers,
    n_features=num_features,
    n_classes=num_classes,
    non_linearity=jax.nn.softmax
)

# train model
Xdata = X.reshape(num_nodes, num_features)
sgc, sgc_loss_list = train(
    sgc,
    optax.adabelief(eta),
    (A_train, Xdata, y),
    train_idx,
    num_reps=num_reps
)

# evaluate model
train_acc = accuracy(sgc, (A_train, Xdata, y), train_idx)
print(f'Train accuracy: {train_acc:.4f}')

if not inductive:
    test_acc = accuracy(sgc, (A_train, Xdata, y), test_idx)
    test_accs = [test_acc]
    print(f'Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}')
else:
    test_accs = []
    for ig in test_graphs:
        
        with open(f'{data_dir}sbm_corr={correlation}_graphnum={ig}.pkl', 'rb') as f:
            test_dataset = pickle.load(f)

        ytest = test_dataset['data']['y']
        
        Adense = test_dataset['data']['A']
        A_test = sparse.BCOO.fromdense(Adense)
        
        num_classes_test = test_dataset['params']['num_classes']
        num_features_test = test_dataset['params']['num_features']
        num_nodes_test = test_dataset['params']['num_nodes']
        Xtest = test_dataset['data']['muX'].reshape(num_nodes_test, num_features_test)
        
        test_idx = jnp.arange(num_nodes_test)
        test_acc = accuracy(sgc, (A_test, Xtest, ytest), test_idx)
        test_accs.append(test_acc)

        print(f'Test accuracy on graph {ig}: {test_acc:.6f}')


# save trained model
if save_result:
    eqx.tree_serialise_leaves(f'{result_dir}{correlation}/weights_model=sgc_corr={correlation}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.eqx', sgc)

# save parameters
param_dict = {'params' : {
                'correlation' : correlation,
                'model' : 'sgc',
                'seed' : seed,
                'num_nodes' : num_nodes,
                'num_features' : num_features,
                'num_classes' : num_classes,
                'num_hidden' : num_hidden,
                'num_layers' : num_layers,
                'train_frac' : float(train_idx.shape[0])/num_nodes,
                'test_frac' : float(test_idx.shape[0])/num_nodes,
                'test_acc' : np.mean(test_accs),
                'train_acc' : train_acc,
                'learning_rate' : eta,
                'num_reps' : num_reps,
                }
            }

if save_result:
    with open(f'{result_dir}{correlation}/hyperparams_model=sgc_corr={correlation}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'wb') as f:

        pickle.dump(param_dict, f)



