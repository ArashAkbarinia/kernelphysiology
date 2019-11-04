"""
Wrappers around different libraries to plot beautiful bar charts!
"""

import numpy as np
from matplotlib import pyplot as plt


def get_cmap(n, name='gist_rainbow'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard colormap name."""
    return plt.cm.get_cmap(name, n)


def plot_list_data(list_data, figsize=(12, 4), width=0.2,
                   colours=None, lines=None,
                   fontsize=14, fontweight='bold',
                   xlabel=None, xticklabels=None,
                   ylabel=None, ylim=None,
                   legend=None, loc='best',
                   plot_median=False, plot_mean=False,
                   save_name=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    num_groups = len(list_data)
    if num_groups % 2 == 0:
        half = (num_groups / 2) - 1
        sp = -0.5 - half
        ep = -sp
        positions = [*np.arange(sp, 0, 1), *np.arange(ep, 0, -1)[::-1]]
    else:
        half = ((num_groups - 1) / 2) - 1
        sp = -1 - half
        ep = -sp
        positions = [*np.arange(sp, 0, 1), 0, *np.arange(ep, 0, -1)[::-1]]
    items_per_group = len(list_data[0])
    # the x locations for the groups
    bar_inds = np.arange(items_per_group)

    if colours is None:
        tmp_colours = get_cmap(num_groups)
        colours = []
        for i in range(num_groups):
            colours.append(tmp_colours(i))

    for i, current_data in enumerate(list_data):
        color = colours[i]
        if lines is None:
            line = ':'
        elif type(lines) is list:
            line = lines[i]
        else:
            line = lines[0]
        current_data = np.array(current_data)
        current_pos = bar_inds + (width * positions[i])
        ax.bar(current_pos, current_data, width, color=color)
        if plot_median:
            ax.axhline(y=np.median(current_data), linestyle=line, color=color)
        if plot_mean:
            ax.axhline(y=np.mean(current_data), linestyle=line, color=color)

    # x-axis
    ax.set_xticks(np.arange(items_per_group))
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    # y-axis
    if ylim is not None:
        ax.set_ylim(ylim)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight=fontweight)

    # legends
    if legend is not None:
        ax.legend(
            legend, prop={'weight': fontweight, 'size': fontsize}, loc=loc
        )

    # if to be saved
    if save_name is not None:
        fig.tight_layout()
        plt.savefig(save_name)
    plt.show()
