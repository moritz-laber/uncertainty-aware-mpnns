"""
plot_radii_synthetic.py

Plot for comparing robustness radii on synthetic data.

M. Laber, 2026/02
"""

## IMPORTS ##
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

## PARAMETERS ##
savefig = True                          # whether to save the figure
input_dir = './experiments/synthetic/'  # directory to load results
output_dir = './plots/'                 # directory to save results 
l = 3                                  # number of layers
corr = 'dndf'                          # correlation name
h = 16                                 # number of hidden units
r = 1000                               # number of training repetitions

figsize = (4.3, 4.3)   # figure size
nbins = 12             # number of histogram bins
title_fs = 10          # title font size
tick_fs = 10           # tick font size
label_fs = 12          # axis label font size
legend_fs = 8          # legend font size
letter_fs = 13         # subplot letter font size

## CONSTANTS ##
nonlinearities = ['relu', 'gelu', 'tanh', 'sigmoid']
letter = ['(a)', '(b)', '(c)', '(d)']
nonlin_dict = {'relu' : r'$\mathtt{ReLU}$', 'gelu' : r'$\mathtt{GELU}$', 'tanh' : r'$\mathtt{tanh}$', 'sigmoid' : r'$\mathtt{sigmoid}$'}
radius_norm = 'rel'             
estimator = 'taylor_uni_quad_gc'
color_dict = {
    'tanh' : '#ff474c',
    'relu' : '#0485d1', 
    'gelu' : '#40a368',
    'sigmoid' : '#f2ab15'
}

## MAIN ##

# load data
results = {}
coverage = {}
lipschitz = {}
time = {}
for nonlin in nonlinearities:

    with open(f'{input_dir}{corr}/radii_model=gcn_corr={corr}_nonlin={nonlin}_l={l}_hidden={h}_reps={r}.pkl', 'rb') as f:
        result = pickle.load(f)

    results[nonlin] = result[estimator]['radii_worstcase']/result[estimator]['lengths']
    coverage[nonlin] = result[estimator]['positive_frac']
    lipschitz[nonlin] = result[estimator]['C_worstcase']
    time[nonlin] = result[estimator]['time_radii'] + result[estimator]['time_moment'] + result[estimator]['time_C_wc']

# plotting
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
    ax.ticklabel_format(style='sci', axis='x', scilimits=(-4,0))
    ax.tick_params(labelsize=tick_fs)
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
    fig.savefig(f'{output_dir}pdf/radii_corr={corr}_l={l}_num_hidden={h}_estim={estimator}.pdf')