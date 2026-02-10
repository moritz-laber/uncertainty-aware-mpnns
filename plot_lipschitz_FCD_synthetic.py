"""
plot_lipschitz_FCD_synthetic.py

Plots for showing Lipschitzness of SGC in FCD on synthetic data. 

M. Laber, 2026/02
"""

### IMPORTS ###
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pickle

### PARAMETERS ###
figsize = (4.2, 2.3)  # figure size
alpha = 0.15          # marker transparency
marker = "."          # marker type
xmax = 13             # x-axis limit
ymax = 13             # y-axis limit
title_fs = 12         # title font size
annotation_fs = 10    # annotation font size
text_fs = 7           # text font size
n_select = 10_000     # number of points to plot to reduce filesize
r = 1000              # number of training repetitions
l = 3                 # number of layers
savefig = True        # whether to save the figure

input_dir = './experiments/synthetic/' # directory from which to load results
output_dir = './plots/'                # directory to save plots


### CONSTANTS ###
letter = {
    (1,0) : '(a)',
    (1,1) : '(b)',
    (1,2) : '(c)'
}

corr_dict = {
    'inif' : r'$\mathtt{inif}$',
    'indf'  : r'$\mathtt{indf}$',
    'dndf' : r'$\mathtt{dndf}$'
}

row = 1


### AUXILIARY FUNCTIONS ###
def z_score(data):

    data = np.asarray(data)
    
    return (data - np.nanmean(data))/np.nanstd(data)


## MAIN ##

# create plot
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(2, 3, height_ratios=[0.01, 1], wspace=0.1, width_ratios=[1, 1, 1], figure=fig)
    
for col, corr in zip([0,1,2], ['inif', 'indf', 'dndf']):

    # load data
    with open(f'{input_dir}{corr}/generalization_model=sgc_corr={corr}_l={l}_reps={r}.pkl', 'rb') as f:
        result = pickle.load(f)

    # title
    title_ax = fig.add_subplot(gs[0, col])
    title_ax.axis('off')
    title_ax.text(0.35, 0.0, corr_dict[corr], fontsize=title_fs, fontweight='bold')

    # plot
    ax = fig.add_subplot(gs[row, col])
    if result['dist_XS'].shape[0] > n_select:
        select = np.random.choice(result['dist_XS'].shape[0], n_select , replace=False)
    else:
        select = np.arange(result['dist_XS'].shape[0])
    ax.plot(z_score(result['dist_XS'])[select], z_score(result['dist_logits'])[select], marker=marker, alpha=alpha, color='#015482')
    ax.set_xlim(-2., 8)
    ax.set_ylim(-2., 8)
    ax.text(
        0.97,
        0.95,
        r'$\rho_\mathrm{p}=$'+f'{result['pearson']:.2f}\n'+r'$\rho_\mathrm{s}=$'+f'{result['spearman']:.2f}\n'+r'$C=$'+f'{result['lipschitz_FCD']:.0f}',
        transform=ax.transAxes,
        fontsize=text_fs,
        va='top',
        ha='right',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.15",))
    ax.text(
        0.05,
        0.95,
        letter[(row,col)],
        transform=ax.transAxes,
        fontsize=10,
        fontweight='bold',
        va='top',
        ha='left')
    if row==1:
        ax.set_xlabel(r'$z(\mathrm{FCD}_1)$')
    else:
        ax.set_xticks([])
    if col==0:
        ax.set_ylabel(r'$z(W_1)$')
    else:
        ax.set_yticks([])

fig.subplots_adjust(left=0.15, bottom=0.29, right=0.95)
fig.savefig(f'{output_dir}synthetic_lipschitz-l={l}.pdf')