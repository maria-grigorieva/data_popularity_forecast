import matplotlib.pyplot as plt
import numpy as np


def visualize_random_sequences(data: np.ndarray, nrows: int, ncols: int) -> tuple:
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_figheight(6 * nrows)
    fig.set_figwidth(24)
    max_y = -1
    for i in range(nrows):
        for j in range(ncols):
            sample_idx = np.random.randint(0, len(data))
            max_y = max(max_y, data[sample_idx].max())
            ax[i, j].bar(np.arange(data[sample_idx].shape[0]), data[sample_idx])
            ax[i, j].set_title(f"{sample_idx} sequence")
    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].set_ylim(0, max_y)
    return fig, ax
    