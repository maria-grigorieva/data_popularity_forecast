import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# Custom colormap for forthcoming HeatMap
# Blue-to-Red, but with White zeros

newcolors = np.zeros((256, 4))
for i in range(newcolors.shape[0]):
    newcolors[i] = [i / 255, 0.0, (newcolors.shape[0] - i - 1) / 255, 1.0]
newcolors[0] = [1.0, 1.0, 1.0, 1.0]
newcm = ListedColormap(newcolors)

def plot_heatmap_accesses(
        heat_matrix,
        axis=None,
        title=None,
        savefig_path=None
):

    # heat_matrix = np.stack(objs_df['history_ts'].to_numpy())

    if axis:
        ax = axis
    else:
        fig, ax = plt.subplots()
        fig.set(figwidth=16, figheight=6)
    ax.set(xlabel='day #', ylabel='dataset #')

    if title:
        ax.set(title=title)

    psm = ax.pcolormesh(heat_matrix, cmap=newcm, rasterized=True)
    if axis is None:
        fig.colorbar(psm, ax=ax)

    if (axis is None) and savefig_path:
        fig.savefig(savefig_path, bbox_inches='tight')

def plot_comparison(df1, df2, savefig_path=None):

    fig, axes = plt.subplots(3, 2)
    fig.set(figwidth=32, figheight=18)
    plot_heatmap_accesses(np.stack(df1['history_ts'].to_numpy()),
                          axis=axes[0, 0],
                          title='Access heatmap (without tid merging)')
    plot_heatmap_accesses(np.stack(df1[df1['y'] > 0]['history_ts'].to_numpy()),
                          axis=axes[1, 0], title='Positive datasets only')
    plot_heatmap_accesses(np.stack(df1[df1['y'] == 0]['history_ts'].to_numpy()),
                          axis=axes[2, 0], title='Negative datasets only')

    plot_heatmap_accesses(np.stack(df2['history_ts'].to_numpy()),
                          axis=axes[0, 1],
                          title='Access heatmap (with tid merging)')
    plot_heatmap_accesses(np.stack(df2[df2['y'] > 0]['history_ts'].to_numpy()),
                          axis=axes[1, 1], title='Positive datasets only')
    plot_heatmap_accesses(np.stack(df2[df2['y'] == 0]['history_ts'].to_numpy()),
                          axis=axes[2, 1], title='Negative datasets only')

    if savefig_path:
        fig.savefig(savefig_path, bbox_inches='tight')
