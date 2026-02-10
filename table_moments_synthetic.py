"""
table_moments_synthetic.py

Table for comparing moment propagation methods on synthetic data.

M. Laber, 2026/02
"""

## IMPORTS ##
import pickle
import pandas as pd

## PARAMETERS ##
savetable = False                        # whether to save the table
input_dir = './experiments/synthetic/'   # directory to load results
output_dir = './tables/'                 # directory to save results
h = 16     # number of hidden units
r = 1000   # number of training repetitions


## MAIN ##
results = []
for corr in ['inif', 'indf', 'dndf']:
    for l in [2,3]:
        print(f'getting: l={l}, corr={corr}', end='\r')
        for nonlin in ['relu', 'gelu', 'tanh', 'sigmoid']:
            
            success = False
            try:
                with open(f'{input_dir}{corr}/comparison_model=gcn_corr={corr}_nonlin={nonlin}_l={l}_hidden={h}_reps={r}.pkl', 'rb') as f:
                    result = pickle.load(f)
                success = True
            except:
                continue

            if success:
                with open(f'{input_dir}{corr}/moments_model=gcn_corr={corr}_nonlin={nonlin}_l={l}_hidden={h}_reps={r}.pkl', 'rb') as f:
                    moments = pickle.load(f)

            if success:
                for i, estimator in enumerate(['taylor_multi_lin', 'taylor_multi_quad', 'ptpe', 'taylor_uni_lin', 'taylor_uni_quad', 'taylor_uni_quad_gc']):
                    row = [corr, l, h, nonlin, estimator]
                    row.append(result['results']['fro_mu'][i])
                    row.append(result['results']['fro_sig'][i])
                    row.append(result['results']['wasserstein'][i])
                    row.append(moments['moments'][estimator]['time'])
                    results.append(row)
                row = [corr, l, h, nonlin, 'sampling']
                row += 3*[pd.NA]
                row.append(moments['moments']['sample']['time'])
                results.append(row)
            else:
                for i, estimator in enumerate(['taylor_multi_lin', 'taylor_multi_quad', 'ptpe', 'taylor_uni_lin', 'taylor_uni_quad', 'taylor_uni_quad_gc']):
                    row = [corr, l, h, nonlin, estimator]
                    row += 4*[pd.NA]
                    results.append(row)
                row = [corr, l, h, nonlin, 'sampling']
                row += 4*[pd.NA]
                results.append(row)

result_df = pd.DataFrame(results, columns=['corr', 'num_layers', 'num_hidden', 'nonlin', 'estimator', 'fro_mu', 'fro_sig', 'wasserstein', 'time'])

if savetable:
    result_df.to_csv(f'{output_dir}moment_comparison.csv', index=False)