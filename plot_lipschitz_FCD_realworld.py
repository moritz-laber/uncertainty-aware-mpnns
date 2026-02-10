
"""
plot_lipschitz_FCD_realworld.py

Plots for showing Lipschitzness of SGC in FCD on real-world data. 

M. Laber, 2026/02
"""

### IMPORTS ###
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pickle   

### PARAMETERS ###
figsize = (7.8, 1.6)  # figure size
alpha = 0.15          # marker transparency
marker = "."          # marker type
xmax = 25             # x-axis limit
ymax = 25             # y-axis limit
title_fs = 10         # title font size
annotation_fs = 10    # annotation font size
text_fs = 6           # text font size
n_select = 10_000     # number of points to plot to reduce filesize
wspace=0.15           # width space between subplots
l = 3                 # number of layers
savefig = True        # whether to save the figure

input_dir = './experiments/realworld/' # directory from which to load results
output_dir = './plots/'                # directory to save plots

### CONSTANTS ###
letter = {
    (1,0) : '(a)',
    (1,1) : '(b)',
    (1,2) : '(c)',
    (1,3) : '(d)',
    (1,4) : '(e)',
    (1,5) : '(f)',
    (1,6) : '(g)',
    (2,0) : '(h)',
    (2,1) : '(i)',
    (2,2) : '(j)',
    (2,3) : '(k)',
    (2,4) : '(l)',
    (2,5) : '(m)',
    (2,6) : '(n)'
}

row = 1

## AUXILIARY FUNCTIONS ##
def z_score(data):

    data = np.asarray(data)
    
    return (data - np.nanmean(data))/np.nanstd(data)


### MAIN ###

# create plot
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(2, 7, height_ratios=[0.05, 1], width_ratios=np.ones(7), wspace=wspace, figure=fig)

for col, dataset, datasetstr in zip([0, 1, 2, 3, 4, 5, 6], ['cornell', 'texas', 'wisconsin', 'cora', 'citeseer', 'chameleon', 'squirrel'], [r'$\mathtt{cornell}$', r'$\mathtt{texas}$', r'$\mathtt{wisconsin}$', r'$\mathtt{cora}$', r'$\mathtt{citeseer}$', r'$\mathtt{chameleon}$', r'$\mathtt{squirrel}$']):

    # load data
    for r in [125, 250, 500, 1000, 2000, 4000]:
        try:
            with open(f'{input_dir}{dataset}/generalization_model=sgc_data={dataset}_l={l}_reps={r}.pkl', 'rb') as f:
                result = pickle.load(f)
            break
        except:
            pass

    # title
    title_ax = fig.add_subplot(gs[0, col])
    title_ax.axis('off')
    title_ax.text(0.5, 0, datasetstr, fontsize=title_fs, fontweight='bold', va='center', ha='center')

    # plot
    ax = fig.add_subplot(gs[row, col])
    if result['dist_XS'].shape[0] > n_select:
        select = np.random.choice(result['dist_XS'].shape[0], n_select , replace=False)
    else:
        select = np.arange(result['dist_XS'].shape[0])
    ax.scatter(z_score(result['dist_XS'])[select], z_score(result['dist_logits'])[select], marker=marker, alpha=alpha, color='#015482')
    ax.set_xlim(-4, 11)
    ax.set_ylim(-3, 11)
    if col==0:
        ax.set_yticks([0, 4, 8])
    else:
        ax.set_yticks([])
    ax.tick_params(labelsize=10)
    ax.text(
        0.95,
        0.95,
        r'$\rho_\mathrm{p}=$'+f'{result["pearson"]:.2f}\n'+r'$C=$'+f'{result["lipschitz_FCD"]:.0f}',
        transform=ax.transAxes,
        fontsize=text_fs,
        va='top',
        ha='right',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.15",))
    ax.text(
        0.04,
        0.95,
        letter[(row,col)],
        transform=ax.transAxes,
        fontsize=9,
        va='top',
        ha='left',
        fontweight='bold')
    if row == 1:
        ax.set_xlabel(r'$z(\mathrm{FCD}_1)$', fontsize=10)
        ax.set_xticks([-4, 0, 4, 8])
    if row == 2:
        ax.set_xticks([])
    if col==0:
        ax.set_ylabel(r'$z(W_1)$', fontsize=10)

fig.subplots_adjust(bottom=0.35, top=0.95, wspace=wspace)
if savefig: fig.savefig(f'{output_dir}realworld_lipschitz_l={l}.pdf')