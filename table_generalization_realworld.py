"""
table_generalization_realworld.py
----------

Creates a table with generalization results from
experiments on realworld data.
"""

### IMPORTS ###
import matplotlib.pyplot as plt
import pickle 
import numpy as np
import jax
import jax.numpy as jnp
import os
import pandas as pd

### Parameters ###
savetable = False                      # whether to save the table
input_dir = './experiments/realworld/' # directory with results
output_dir = './plots/generalization/' # directory for saving table 
datasets = [                           # list of datasets to consider
    'cornell',
    'wisconsin',
    'texas',
    'cora',
    'citeseer',
    'chameleon',
    'squirrel'
    ]
layers = [2,3]        # number of layers
reps =  [             # number of repetitions to try
    125,
    250,
    500,
    1000,
    2000,
    4000
    ]

result_df = []
for dataset in datasets:
    for l in layers:
        # load data
        success = False
        for r in reps:
            try:
                with open(f'{input_dir}{dataset}/multiclass_generalization_model=sgc_data={dataset}_l={l}_reps={r}.pkl','rb') as f:
                    result = pickle.load(f)
                success = True
                break
            except:
                pass
        
        # create row in table
        if success:
            result_df.append([dataset,
                              l,
                              np.max(result['empirical_LHS']),
                              result['bound'],              # we used the wrong Cl in experiments
                              result['lipschitz_FCD'],
                              result['loss_lipschitz_l2'],      # we used the wrong Cl in experiments
                              result['M'],
                              result['epsilon'],
                              result['delta'],
                              result['spearman'],
                              result['pearson'],
                              result['K'],
                              result['Chi'],
                             ])
        else:
            row = [dataset, l] + 10 * [pd.NA]
            result_df.append(row)

# creat table from rows
result_df = pd.DataFrame(result_df, columns=['dataset',
                                             'num_layers',
                                             'empirical_LHS',
                                             'upper_bound',
                                             'lipschitz_FCD',
                                             'lipschitz_loss',
                                             'M',
                                             'epsilon',
                                             'delta',
                                             'spearman',
                                             'pearson',
                                             'K',
                                             'Chi'])
# save table
if savetable:
    result_df.to_csv(f'{output_dir}/generalization_realworld.csv', index=False)
    result_df.to_latex(f'{output_dir}/generalization_realworld.tex', index=False, float_format="%.2f")