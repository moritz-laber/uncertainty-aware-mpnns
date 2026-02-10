"""
table_generalization_synthetic.py
----------

Creates a table with generalization results from
experiments on synthetic data.

M. Laber, 2026/02
"""

### IMPORTS ###
import pickle
import pandas as pd
import numpy as np

### PARAMETERS ###
savetable = True                        # whether to save the table
input_dir = './experiments/synthetic/'  # directory with results
output_dir = './tables/'                # directory to save the table
r = 1000                                # number of training repetitions

### MAIN ###
result_df = []
for corr in ['inif', 'indf', 'dndf']:
    for l in [2,3]:

        success = False
        try:
            with open(f'{input_dir}{corr}/generalization_model=sgc_corr={corr}_l={l}_reps={r}.pkl','rb') as f:
                result = pickle.load(f)
            success = True
        except:
            continue

        if success:
            result_df.append([corr,
                              l,
                              np.max(result['empirical_LHS']),
                              result['bound'],
                              result['lipschitz_FCD'],
                              result['loss_lipschitz_l2'], 
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

# create the table
result_df = pd.DataFrame(result_df, columns=['corr',
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

if savetable:
    result_df.to_csv(f'{output_dir}generalization_synthetic.csv', index=False)