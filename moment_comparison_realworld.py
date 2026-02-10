"""
moment_comparison_realworld.py
----------

Compares moments obtained on real-world data.

M. Laber, 2026/02
"""

### IMPORTS ###
import jax.numpy as jnp
import pickle
import sys

from models import *
from utils import *

### PARAMETERS ###
input_dir = './experiments/realworld/'    # directory to load results
output_dir = './experiments/realworld/'   # directory to save results

dataset = sys.argv[1]                  # dataset name
num_layers = int(sys.argv[2])          # number of GCN layers
num_hidden = int(sys.argv[3])          # number of hidden units
num_reps = int(sys.argv[4])            # training reps
nonlin_str = sys.argv[5]               # nonlinearity
diagW2 = bool(int(sys.argv[6]))        # whether to use full or diagonal W2 distance
eps = 1e-15                            # small constant for numerical stability in matrix square root

### MAIN ###

## Load Moments ##
with open(f'{input_dir}{dataset}/moments_model=gcn_data={dataset}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'rb') as f:

    moments = pickle.load(f)

muY_sample = moments['moments']['sample']['mu']
SigY_sample = moments['moments']['sample']['Sig']
muY_lin = moments['moments']['taylor_multi_lin']['mu']
SigY_lin = moments['moments']['taylor_multi_lin']['Sig']
muY_quad = moments['moments']['taylor_multi_quad']['mu']
SigY_quad = moments['moments']['taylor_multi_quad']['Sig']
muY_ptpe = moments['moments']['ptpe']['mu']
SigY_ptpe = moments['moments']['ptpe']['Sig']
muY_tay1 = moments['moments']['taylor_uni_lin']['mu']
SigY_tay1 = moments['moments']['taylor_uni_lin']['Sig']
muY_tay2 = moments['moments']['taylor_uni_quad']['mu']
SigY_tay2 = moments['moments']['taylor_uni_quad']['Sig']
muY_tay2_gc = moments['moments']['taylor_uni_quad_gc']['mu']
SigY_tay2_gc = moments['moments']['taylor_uni_quad_gc']['Sig']

## Analysis ##
print('Analysis')

# mean
print('relative error mean')
relative_errors_mu = [
    relative_error(muY_lin, muY_sample),
    relative_error(muY_quad, muY_sample),
    relative_error(muY_ptpe, muY_sample),
    relative_error(muY_tay1, muY_sample),
    relative_error(muY_tay2, muY_sample),
    relative_error(muY_tay2_gc, muY_sample)
    ]

print('absolute error mean')
fro_errors_mu = [
    jnp.linalg.norm(muY_lin - muY_sample),
    jnp.linalg.norm(muY_quad - muY_sample),
    jnp.linalg.norm(muY_ptpe - muY_sample),
    jnp.linalg.norm(muY_tay1 - muY_sample),
    jnp.linalg.norm(muY_tay2 - muY_sample),
    jnp.linalg.norm(muY_tay2_gc - muY_sample)
    ]

# Covariance
print('relative error covariance')
relative_errors_sig = [
    relative_error(SigY_lin, SigY_sample),
    relative_error(SigY_quad, SigY_sample),
    relative_error(SigY_ptpe, SigY_sample),
    relative_error(SigY_tay1, SigY_sample),
    relative_error(SigY_tay2, SigY_sample),
    relative_error(SigY_tay2_gc, SigY_sample)
    ]

print('absolute error covariance')
fro_errors_sig = [
        jnp.linalg.norm(SigY_lin - SigY_sample),
        jnp.linalg.norm(SigY_quad - SigY_sample),
        jnp.linalg.norm(SigY_ptpe - SigY_sample),
        jnp.linalg.norm(SigY_tay1 - SigY_sample),
        jnp.linalg.norm(SigY_tay2 - SigY_sample),
        jnp.linalg.norm(SigY_tay2_gc - SigY_sample)
        ]

# Wasserstein 2
print('Wasserstein distance')
print('lin')
w2_lin = wasserstein_distance((muY_lin, SigY_lin), (muY_sample, SigY_sample), eps=eps, diag=diagW2)
print('quad')
w2_quad = wasserstein_distance((muY_quad, SigY_quad), (muY_sample, SigY_sample), eps=eps, diag=diagW2)
print('ptpe')
w2_ptpe = wasserstein_distance((muY_ptpe, SigY_ptpe), (muY_sample, SigY_sample), eps=eps, diag=diagW2)
print('lin')
w2_tay1 = wasserstein_distance((muY_tay1, SigY_tay1), (muY_sample, SigY_sample), eps=eps, diag=diagW2)
print('quad')
w2_tay2 = wasserstein_distance((muY_tay2, SigY_tay2), (muY_sample, SigY_sample), eps=eps, diag=diagW2)
print('quad gc')
w2_tay2_gc = wasserstein_distance((muY_tay2_gc, SigY_tay2_gc), (muY_sample, SigY_sample), eps=eps, diag=diagW2)

wasserstein_dist_result = [  
        w2_lin,
        w2_quad,
        w2_ptpe,
        w2_tay1,
        w2_tay2,
        w2_tay2_gc
        ]

## Save Results ###
moments['params']['eps'] = eps

result_dict = {'results': {
                    'rel_mu' : relative_errors_mu,
                    'fro_mu' : fro_errors_mu,
                    'rel_sig' : relative_errors_sig,
                    'fro_sig' : fro_errors_sig,
                    'wasserstein' : wasserstein_dist_result,
                    'order' :  ['all_lin', 'all_quad_trunc', 'ptpe', 'layer_lin', 'layer_quad_trunc', 'layer_quad_gc'],
                     },
               'params' : {k : v for k,v in moments['params'].items()}
                }


with open(f'{output_dir}{dataset}/comparison_model=gcn_data={dataset}_nonlin={nonlin_str}_l={num_layers}_hidden={num_hidden}_reps={num_reps}.pkl', 'wb') as f:

    pickle.dump(result_dict, f)



