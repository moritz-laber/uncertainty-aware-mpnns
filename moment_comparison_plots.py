"""
moment_comparison_plots.py
----------

Plots for the comparison of different moment propagation methods for MPNNs.
"""

import pickle
from unittest import result
import numpy as np
import matplotlib.pyplot as plt

### PARAMETERS ###
result_dir = './results/moment_comparison/'
plot_dir = './plots/moment_comparison/'

distance = 'wasserstein'        # distance: Wasserstein, rel_mu, rel_sig, fro_mu, fro_sig
correlation = 'dndf'            # dndf, indf, inif
fs = 12                         # font size
width = 0.2                     # width of bars
rot = 60                        # rotation of x-ticks
ymax = 0.95                     # upper limit y-axis
savefig = True                  # whether to save the figure

### MAIN ###

# load data
result = {}
for nonlin_str in ['relu', 'tanh', 'gelu']:
    with open(f'{result_dir}comparison_corr={correlation}_fun={nonlin_str}.pkl', 'rb') as f:

        result[nonlin_str] = pickle.load(f)




w_list = [np.asarray(result[nonlin_str]['results'][distance]) for nonlin_str in ['tanh', 'relu', 'gelu']]

# create plot
fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111)

for w, dx, c, l in zip(w_list, [-width, 0., width], ['#ff474c','#0485d1','#40a368'], ['tanh', 'relu', 'gelu']):

    w = np.delete(w, 2)
    ax.bar(np.arange(w.shape[0])-dx, w, color=c, label=l, width=width)

ax.legend()
ax.set_xticks(np.arange(w.shape[0]), [r'$\mathtt{T1}$',
                                      r'$\mathtt{T2}$'+'-'+r'$\mathtt{Tr}$',
                                      r'$\mathtt{PTPE}$',
                                      r'$1$d'+'-'+r'$\mathtt{T1}$',
                                      r'$1$d'+'-'+r'$\mathtt{T2}$'+'-'+r'$\mathtt{Tr}$',
                                      r'$1$d'+'-'+r'$\mathtt{T2}$'+'-'+r'$\mathtt{GC}$'
                                     ], rotation=rot)

ax.set_ylabel(r'$W_2(\mathcal{N}(\mu, \Sigma),\mathcal{N}(\mu_\mathrm{sample},\Sigma_\mathrm{sample}$)', fontsize=fs)
ax.tick_params(labelsize=fs-2)
ax.set_ylim(0., ymax)
fig.tight_layout()

# save figure
if savefig:
    fig.savefig(f'{plot_dir}comparison_W2_{correlation}.png', dpi=600)
    fig.savefig(f'{plot_dir}comparison_W2-{correlation}.pdf')
    fig.savefig(f'{plot_dir}comparison_W2-{correlation}.svg')