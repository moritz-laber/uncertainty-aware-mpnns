"""
table_generalization_synthetic.py
----------

Creates a table with generalization results from
experiments on synthetic data.
"""

### IMPORTS ###
import matplotlib.pyplot as plt
import pickle 
import numpy as np
import jax
import jax.numpy as jnp
import os
import pandas as pd

### PARAMETERS ###
savetable = False                        # whether to save the table
input_dir = './experiments/synthetic/'   # directory with results
output_dir = './plots/generalization/'   # directory for saving table 
r = 1000                                 # number of training repetitions
layers = [2, 3]                          # number of layers
correlations = ['inif', 'indf', 'dndf']  # correlations

result_df = []
for corr in correlations:
    for l in layers:
        # load data
        success = False
        try:
            with open(f'{input_dir}{corr}/generalization_model=sgc_corr={corr}_l={l}_reps={r}.pkl','rb') as f:
                result = pickle.load(f)
            success = True
        except:
            continue

        # create a row in the table
        if success:
            result_df.append([corr,
                              l,
                              np.max(result['empirical_LHS']),
                              result['bound'],
                              result['lipschitz_FCD'],
                              result['M'],
                              result['epsilon'],
                              result['delta'],
                              result['spearman'],
                              result['pearson'],
                              result['K'],
                              result['Chi']
                             ])
        else:
            row = [corr, l] + 10 * [pd.NA]
            result_df.append(row)

# create the table from rows
result_df = pd.DataFrame(result_df, columns=['corr',
                                             'num_layers',
                                             'empirical_LHS',
                                             'upper_bound',
                                             'lipschitz_FCD',
                                             'M',
                                             'epsilon',
                                             'delta',
                                             'spearman',
                                             'pearson',
                                             'K',
                                             'Chi'])
# save the table
if savetable:
    result_df.to_csv(f'{output_dir}/generalization_synthetic.csv', index=False)
    result_df.to_latex(f'{output_dir}/generalization_synthetic.tex', index=False, float_format="%.2f")