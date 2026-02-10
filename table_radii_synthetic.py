"""
table_radii_synthetic.py

Table for comparing robustness radii on synthetic data.

M. Laber, 2026/02
"""

## IMPORTS ##
import pickle
import numpy as np
import pandas as pd

## PARAMETERS ##
savetable = True                       # whether to save the table  
input_dir = './experiments/synthetic/' # directory to load results
output_dir = './tables/'               # directory to save results
h = 16                                 # number of hidden units
r = 1000                               # number of training repetitions

## MAIN ##
nonlinearities = ['relu', 'gelu', 'tanh', 'sigmoid']
result_df = []
for corr in ['inif', 'indf', 'dndf']:
    for l in [2,3]:
        for estimator in ['sample', 'ptpe', 'taylor_uni_quad_gc', 'taylor_multi_lin']:
            print(f'getting: l={l}, data={corr}, est={estimator}', end='\r')

            for nonlin in nonlinearities:
                try:
                    with open(f'{input_dir}{corr}/radii_model=gcn_corr={corr}_nonlin={nonlin}_l={l}_hidden={h}_reps={r}.pkl', 'rb') as f:
                        result = pickle.load(f)
                    with open(f'{input_dir}{corr}/cohen_radii_model=gcn_corr={corr}_nonlin={nonlin}_l={l}_hidden={h}_reps={r}.pkl', 'rb') as f:
                        result_cohen = pickle.load(f)
                except:
                    continue

                row = [corr, l, h, nonlin, estimator]
                row.append(result[estimator]['positive_frac'])
                row.append(result_cohen['positive_frac'])
                row.append(np.nanmean(result[estimator]['radii_worstcase'][result[estimator]['radii_worstcase']>0]))
                row.append(np.nanmean(result_cohen['radii'][result_cohen['radii']>0]))
                row.append(result[estimator]['C_worstcase'])
                row.append(result[estimator]['time_radii'] + result[estimator]['time_moment'] + result[estimator]['time_C_wc'])
                row.append(result_cohen['time_radii'])
                result_df.append(row)

result_df = pd.DataFrame(result_df, columns=['corr','num_layers', 'num_hidden', 'nonlin', 'estimator', 'certified_ours', 'certified_cohen', 'radii', 'radii_cohen', 'C_lipschitz', 'time_ours', 'time_cohen'])
if savetable:
    result_df.to_csv(f'{output_dir}radii.csv', index=False)
                    