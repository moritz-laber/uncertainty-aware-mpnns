"""
plot_moments_realworld.py

Plots for comparing moment propagation methods on real-world data.

M. Laber, 2026/02
"""

## IMPORT ##
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

### PARAMETERS ###
savefig = True                                     # whether to save the figure
dataset = 'cora'                                   # dataset name
l = 3                                              # number of layers
num_hidden = 8                                     # number of hidden units
input_dir = './experiments/realworld/'             # directory to load results
output_dir = './plots/'                            # directory to save results

figsize = (3.3, 0.8)        # figure size
tick_fs = 12                # tick font size
label_fs = 14               # y-axis label font size
label_fs_2 = 15             # x-axis label font size
legend_fs = 12              # legend font size
title_fs = 14               # title font size
ylim = (1e-2, 5e3)          # y-axis limits
logy = True                 # whether to use log scale on y-axis


## MAIN ##

# constants for plotting
error_type = 'wasserstein'

label_dict = {
    'wasserstein' : r'$W_2(\mathcal{N}, \mathcal{N}^\mathrm{MC})$',
}

color_dict = {
    'tanh' : '#ff474c',
    'relu' : '#0485d1', 
    'gelu' : '#40a368',
    'sigmoid' : '#f2ab15'
    
}

legend_dict = {
    'gelu' : r'$\mathtt{GELU}$',
    'relu' : r'$\mathtt{ReLU}$',
    'tanh' : r'$\mathtt{tanh}$',
    'sigmoid' : r'$\mathtt{sigmoid}$'
}

# load data
filelist = os.listdir(f'{input_dir}{dataset}/')
results = {}
for file in filelist:
    if 'comparison' in file and f'l={l}' in file and f'hidden={num_hidden}' in file:
        with open(f'{input_dir}{dataset}/{file}', 'rb') as f:
            result = pickle.load(f)
        nonlin = result['params']['nonlinearity']
        error = result['results'][error_type]
        results[nonlin] = error

# create plot
fig = plt.figure(figsize=(7,3))
ax = fig.add_subplot(111)

for i, nonlin_result in enumerate(results.items()):
    nonlin, result = nonlin_result
    ax.bar(1.5*np.arange(0, len(result))+(0.25*i - 0.5), result, label=legend_dict[nonlin], width=0.24, align='edge', color=color_dict[nonlin])

ax.set_xticks(1.5*np.arange(0, len(result)))
ax.set_xticklabels([r'$\mathtt{T1}$', r'$\mathtt{T2}$-$\mathtt{Tr}$', r'$\mathtt{PTPE}$', r'$\mathtt{1d}$-$\mathtt{T1}$', r'$\mathtt{1d}$-$\mathtt{T2}$-$\mathtt{Tr}$', r'$\mathtt{1d}$-$\mathtt{T2}$-$\mathtt{GC}$'], rotation=0)
if logy: ax.set_yscale('log')
ax.tick_params(axis='y', labelsize=tick_fs)
ax.tick_params(axis='x', labelsize=label_fs_2)
ax.legend(fontsize=legend_fs, loc='upper left', ncol=4)
ax.set_ylim(*ylim)
ax.set_ylabel(label_dict[error_type], fontsize=label_fs)
fig.tight_layout()

if savefig:
    fig.savefig(f'{output_dir}pdf/moment_comparrison_data={dataset}_l={l}_num_hidden={num_hidden}_err={error_type}.pdf')