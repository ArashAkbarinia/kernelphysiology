"""
Wrapper to plot distribution in beautiful formats!
"""

from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter


def plot_violinplot(list_data, figsize=(6, 4),
                    face_colours=None, edge_colours=None,
                    fontsize=14, fontweight='bold', rotation=0,
                    xlabel=None, xticklabels=None,
                    xminortick=None, xminorticklabels=None,
                    ylabel=None, ylim=None,
                    plot_median=False, plot_mean=False,
                    save_name=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    num_groups = len(list_data)

    violin_parts = ax.violinplot(
        list_data, showmeans=plot_mean, showmedians=plot_median
    )
    # setting colours
    if face_colours is not None:
        for i, pc in enumerate(violin_parts['bodies']):
            color = face_colours[i]
            pc.set_facecolor(color)
    if edge_colours is not None:
        for i, pc in enumerate(violin_parts['bodies']):
            color = edge_colours[i]
            pc.set_edgecolor(color)

    # x-axis
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)
    if xticklabels is not None:
        ax.set_xticks([y + 1 for y in range(num_groups)])
        ax.set_xticklabels(xticklabels, rotation=rotation)
    if xminortick is not None:
        ax.set_xticks(xminortick, minor=True)
        ax.xaxis.set_minor_formatter(FixedFormatter(xminorticklabels))
        ax.xaxis.set_tick_params(bottom=False, which='minor')

    # y-axis
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.yaxis.grid(True)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight=fontweight)

    # if to be saved
    if save_name is not None:
        fig.tight_layout()
        plt.savefig(save_name)
    plt.show()
