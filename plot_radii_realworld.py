"""
plot_radii_realworld.py

Plot for comparing robustness radii on real-world data.

M. Laber, 2026/02
"""

## IMPORTS ##
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

## PARAMETERS ##
savefig = True                          # whether to save the figure
input_dir = './experiments/realworld/'  # directory to load results
output_dir = './plots/'                 # directory to save results



figsize = (4.3, 4.3)
nbins = 12
radius_norm = 'rel'
title_fs = 10
tick_fs = 10
label_fs = 12
legend_fs = 8
letter_fs = 13

### CONSTANTS ###
dataset = 'cora'
l = 3
h = 8
estimator = 'taylor_uni_quad_gc'
letter = ['(a)', '(b)', '(c)', '(d)']
nonlinearities = ['relu', 'gelu', 'tanh', 'sigmoid']
color_dict = {
    'tanh' : '#ff474c',
    'relu' : '#0485d1', 
    'gelu' : '#40a368',
    'sigmoid' : '#f2ab15'
}
nonlin_dict = {'relu' : r'$\mathtt{ReLU}$', 'gelu' : r'$\mathtt{GELU}$', 'tanh' : r'$\mathtt{tanh}$', 'sigmoid' : r'$\mathtt{sigmoid}$'}


### MAIN ###

# load data:
results = {}
coverage = {}
lipschitz = {}
time = {}
for nonlin in nonlinearities:

    for r in [250, 500, 1000, 2000, 4000]:
        try:
            with open(f'{input_dir}{dataset}/multiclass_radii_model=gcn_data={dataset}_nonlin={nonlin}_l={l}_hidden={h}_reps={r}.pkl', 'rb') as f:
                result = pickle.load(f)
            break
        except:
            continue

    results[nonlin] = result[estimator]['radii_worstcase']/result[estimator]['lengths']
    coverage[nonlin] = result[estimator]['positive_frac']
    lipschitz[nonlin] = result[estimator]['C_worstcase']
    time[nonlin] = result[estimator]['time_radii'] + result[estimator]['time_moment'] + result[estimator]['time_C_wc']

# create plot
fig = plt.figure(figsize=figsize)
axes = [
    fig.add_subplot(221),
    fig.add_subplot(222),
    fig.add_subplot(223),
    fig.add_subplot(224)
]

for i, (ax, nonlin) in enumerate(zip(axes,nonlinearities)):

    res = results[nonlin]
    counts, _, _ = ax.hist(res[res>0], bins=nbins, color=color_dict[nonlin], density=False, rwidth=0.92)

    ax.set_title(nonlin_dict[nonlin], fontsize=title_fs, fontweight='bold')
    ax.text(0.01, 1.05, letter[i], transform=ax.transAxes, fontweight='bold', fontsize=letter_fs)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(-3,0))
    ax.tick_params(labelsize=tick_fs)
    if not np.all(counts == 0):
        ax.set_ylim(0., 1.1*np.max(counts))
    ax.set_xlim(0., None)
    ax.set_xlabel(r'$\epsilon_i / ||x_i||_2$', fontsize=label_fs)
    ax.text(0.97, 0.95, r'$f_c=$'+f'{100*coverage[nonlin]:.0f}% \n'+r'$C=$'+f'{lipschitz[nonlin]:.0f} \n'+r'$T=$'+f'{time[nonlin]:.0f}s',
            transform=ax.transAxes,
            fontsize=legend_fs,
            va='top',
            ha='right',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"),
            multialignment='left')
    if i in [0, 2]:
        ax.set_ylabel('count', fontsize=label_fs)

fig.tight_layout()
if savefig:
    fig.savefig(f'{output_dir}pdf/multiclass_radii_data={dataset}_l={l}_num_hidden={h}_estim={estimator}.pdf')