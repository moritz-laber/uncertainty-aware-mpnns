"""
table_generalization_realworld.py
----------

Creates a table with generalization results from
experiments on realworld data.

M. Laber, 2026/02
"""

### IMPORTS ###
import pickle
import pandas as pd
import numpy as np

### PARAMETERS ###
savetable = True  # whether to save the table
input_dir = './experiments/realworld/'  # directory with results
output_dir = './tables/'                # directory to save the table

### MAIN ###

# load the data
result_df = []
for dataset in ['cornell', 'wisconsin', 'texas', 'cora', 'citeseer', 'chameleon', 'squirrel']:
    for l in [2,3]:
        success = False
        for r in [125, 250, 500, 1000, 2000, 4000]:
            try:
                path_to_file = f'{input_dir}{dataset}/generalization_model=sgc_data={dataset}_l={l}_reps={r}.pkl'
                    
                with open(path_to_file,'rb') as f:
                    result = pickle.load(f)
                success = True
                break

            except:
                pass

        if success:
            result_df.append([dataset,
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
                              result['Chi'],
                             ])
        else:
            row = [dataset, l] + 10 * [pd.NA]
            result_df.append(row)

# create the table
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

if savetable:
    result_df.to_csv(f'{output_dir}generalization_realworld.csv', index=False)