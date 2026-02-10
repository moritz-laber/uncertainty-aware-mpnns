"""
table_radii_realworld.py

Table for comparing robustness radii on real-world data.

M. Laber, 2026/02
"""

## IMPORTS ##
import pickle
import numpy as np
import pandas as pd

## PARAMETERS ##
savetable = True                           # whether to save the table
input_dir = './experiments/realworld/'     # directory to load results
output_dir = './tables/'                   # directory to save results 

## MAIN ##

nonlinearities = ['relu', 'gelu', 'tanh', 'sigmoid']
result_df = []
for dataset in ['cornell', 'wisconsin', 'texas', 'cora', 'citeseer', 'chameleon', 'squirrel']:
    for l in [2,3]:
        if dataset in ['wisconsin', 'texas', 'citeseer', 'squirrel']:
            h = 4
        elif dataset == 'cornell' and l == 2:
            h = 4
        elif dataset == 'cornell' and l == 3:
            h = 8
        elif dataset in ['cora', 'chameleon']:
            h = 8
        else:
            print("Error: Unknown dataset layer combination")

        for estimator in ['sample', 'ptpe', 'taylor_uni_quad_gc', 'taylor_multi_lin']:
            for lipschitz_estimator in ['wc']:
    
                print(f'getting: l={l}, data={dataset}, est={estimator},  C={lipschitz_estimator}', end='\r')
                for nonlin in nonlinearities:
                
                    for r in [250, 500, 1000, 2000, 4000]:
                        try:
                            with open(f'{input_dir}{dataset}/radii_model=gcn_data={dataset}_nonlin={nonlin}_l={l}_hidden={h}_reps={r}.pkl', 'rb') as f:
                                result = pickle.load(f)
                            with open(f'{input_dir}{dataset}/cohen_radii_model=gcn_data={dataset}_nonlin={nonlin}_l={l}_hidden={h}_reps={r}.pkl', 'rb') as f:
                                result_cohen = pickle.load(f)
                            break
                        except:
                            continue

                    row = [dataset, l, h, nonlin, estimator]
                    row.append(result[estimator]['positive_frac'])
                    row.append(result_cohen['positive_frac'])
                    row.append(np.nanmean(result[estimator]['radii_worstcase'][result[estimator]['radii_worstcase']>0]))
                    row.append(np.nanmean(result_cohen['radii'][result_cohen['radii']>0]))
                    row.append(result[estimator]['C_worstcase'])
                    row.append(result[estimator]['time_radii'] + result[estimator]['time_moment'] + result[estimator]['time_C_wc'])
                    row.append(result_cohen['time_radii'])
                    result_df.append(row)

result_df = pd.DataFrame(result_df, columns=['dataset','num_layers', 'num_hidden', 'nonlin', 'estimator', 'certified_ours', 'certified_cohen', 'radii', 'radii_cohen', 'C_lipschitz', 'time_ours', 'time_cohen'])
if savetable:
    result_df.to_csv(f'{output_dir}radii.csv', index=False)
                    