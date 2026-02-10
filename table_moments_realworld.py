"""
table_moments_realworld.py

Table for comparing moment propagation methods on real-world data.

M. Laber, 2026/02
"""

## IMPORTS ##
import pickle
import pandas as pd

## PARAMETERS ##
savetable = True                         # whether to save the table
input_dir = './experiments/realworld/'   # directory to load results
output_dir = './tables/'                 # directory to save results

## MAIN ##
results = []
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

        print(f'getting: l={l}, data={dataset}', end='\r')
        for nonlin in ['relu', 'gelu', 'tanh', 'sigmoid']:
            
            success = False
            for r in [250, 500, 1000, 2000, 4000]:
                try:
                    with open(f'{input_dir}{dataset}/comparison_model=gcn_data={dataset}_nonlin={nonlin}_l={l}_hidden={h}_reps={r}.pkl', 'rb') as f:
                        result = pickle.load(f)
                    success = True
                    break
                except:
                    continue

            if success:
                with open(f'{input_dir}{dataset}/moments_model=gcn_data={dataset}_nonlin={nonlin}_l={l}_hidden={h}_reps={r}.pkl', 'rb') as f:
                    moments = pickle.load(f)

            if success:
                for i, estimator in enumerate(['taylor_multi_lin', 'taylor_multi_quad', 'ptpe', 'taylor_uni_lin', 'taylor_uni_quad', 'taylor_uni_quad_gc']):
                    row = [dataset, l, h, nonlin, estimator]
                    row.append(result['results']['fro_mu'][i])
                    row.append(result['results']['fro_sig'][i])
                    row.append(result['results']['wasserstein'][i])
                    row.append(moments['moments'][estimator]['time'])
                    results.append(row)
                row = [dataset, l, h, nonlin, 'sampling']
                row += 3*[pd.NA]
                row.append(moments['moments']['sample']['time'])
                results.append(row)
            else:
                for i, estimator in enumerate(['taylor_multi_lin', 'taylor_multi_quad', 'ptpe', 'taylor_uni_lin', 'taylor_uni_quad', 'taylor_uni_quad_gc']):
                    row = [dataset, l, h, nonlin, estimator]
                    row += 4*[pd.NA]
                    results.append(row)
                row = [dataset, l, h, nonlin, 'sampling']
                row += 4*[pd.NA]
                results.append(row)

result_df = pd.DataFrame(results, columns=['dataset', 'num_layers', 'num_hidden', 'nonlin', 'estimator', 'fro_mu', 'fro_sig', 'wasserstein', 'time'])

if savetable:
    result_df.to_csv(f'{output_dir}moment_comparison.csv', index=False)