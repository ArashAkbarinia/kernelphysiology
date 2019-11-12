"""
Wrapper to plot distribution in beautiful formats!
"""

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import ticker
from mpl_toolkits.mplot3d import axes3d

from kernelphysiology.utils.matutils import find_nearest_ind


def plot_violinplot(list_data, figsize=(6, 4), baseline=None,
                    face_colours=None, edge_colours=None,
                    fontsize=14, fontweight='bold', rotation=0,
                    xlabel=None, xticklabels=None,
                    xminortick=None, xminorticklabels=None,
                    ylabel=None, ylim=None,
                    plot_median=False, plot_mean=False, plot_extrema=True,
                    save_name=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    num_groups = len(list_data)

    violin_parts = ax.violinplot(
        list_data, showmeans=plot_mean, showmedians=plot_median,
        showextrema=plot_extrema
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

    # the first element of baseline is its value, the second its name
    if baseline is not None:
        baseline_part = ax.axhline(
            y=baseline[0], linestyle=':', linewidth=2, color='r'
        )
        ax.legend(
            [baseline_part], [baseline[1]],
            prop={'weight': fontweight, 'size': fontsize}
        )

    # x-axis
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)
    if xticklabels is not None:
        ax.set_xticks([y + 1 for y in range(num_groups)])
        ax.set_xticklabels(xticklabels, rotation=rotation)
    if xminortick is not None:
        ax.set_xticks(xminortick, minor=True)
        ax.xaxis.set_minor_formatter(ticker.FixedFormatter(xminorticklabels))
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


def plot_dist1_vs_dist2(dist1, dist2, figsize=(4, 4),
                        colour='b', marker='o',
                        fontsize=14, fontweight='bold',
                        xlabel=None, xlim=None,
                        ylabel=None, ylim=None,
                        point_texts=None,
                        save_name=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(dist1, dist2, color=colour, marker=marker)
    min_val = np.minimum(dist1.min(), dist2.min())
    max_val = np.maximum(dist1.max(), dist2.max())
    ds = (min_val, max_val)
    de = (min_val, max_val)
    ax.plot(ds, de, '--k')

    if point_texts is not None:
        for zdir, x, y in zip(point_texts, dist1, dist2):
            ax.text(x, y, zdir, fontsize=10, fontweight=fontweight)

    # x-axis
    if xlim is not None:
        ax.set_xlim(xlim)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)

    # y-axis
    if ylim is not None:
        ax.set_ylim(ylim)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight=fontweight)

    ax.axis('equal')

    # if to be saved
    if save_name is not None:
        fig.tight_layout()
        plt.savefig(save_name)
    plt.show()


def _list_to_xyz(list_data):
    xrange = []
    yrange = []
    zrange = []
    for item in list_data:
        xrange.append(item[0])
        yrange.append(item[1])
        zrange.append(item[2])
    return xrange, yrange, zrange


def _extract_z(list_data, num_samples):
    xrange, yrange, zrange = _list_to_xyz(list_data)
    _, _, zs = _extract_xyz(xrange, yrange, zrange, num_samples)
    return zs


def _extract_xyz(xrange, yrange, zrange, num_samples):
    zbase = np.mean(zrange)
    xarray = np.linspace(np.min(xrange), np.max(xrange), num_samples)
    yarray = np.linspace(np.min(yrange), np.max(yrange), num_samples)
    xs, ys = np.meshgrid(xarray, yarray)

    zs = np.zeros(xs.shape)
    tmp_nums = np.zeros(zs.shape)
    for i in range(len(xrange)):
        col_ind = find_nearest_ind(xarray, xrange[i])
        row_ind = find_nearest_ind(yarray, yrange[i])
        zs[row_ind, col_ind] += zrange[i]
        tmp_nums[row_ind, col_ind] += 1
    tmp_nums[tmp_nums == 0] = 1
    zs /= tmp_nums
    zs[zs == 0] = None
    return xs, ys, zs


def plot_z3d(list_data, figsize=(5, 4), num_samples=10,
             cmap='PiYG_r', interpolation='none',
             fontsize=14, fontweight='bold',
             xlabel=None,
             ylabel=None,
             legend=None, loc='best',
             save_name=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    xrange, yrange, zrange = _list_to_xyz(list_data)
    _, _, zs = _extract_xyz(xrange, yrange, zrange, num_samples)

    cax = ax.matshow(
        zs, cmap=cmap, interpolation=interpolation, aspect='auto',
        extent=[np.min(xrange), np.max(xrange), np.max(yrange), np.min(yrange)]
    )
    fig.colorbar(cax)

    # x-axis
    ax.xaxis.set_ticks_position('bottom')
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)

    # y-axis
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


def plot_wireframe3d(list_data, figsize=(8, 8), num_samples=10,
                     colour='b', view_init=(30, 60),
                     fontsize=14, fontweight='bold',
                     xlabel=None,
                     ylabel=None, ylim=None,
                     zlabel=None,
                     legend=None, loc='best',
                     save_name=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # compute the z baseline, x and y range
    xrange, yrange, zrange = _list_to_xyz(list_data)
    xs, ys, zs = _extract_xyz(xrange, yrange, zrange, num_samples)

    ax.plot_wireframe(xs, ys, zs, color=colour)
    ax.view_init(view_init[0], view_init[1])

    # x-axis
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)

    # y-axis
    ax.set_yticks(np.arange(np.min(yrange), np.max(yrange) + 0.1, 5.0))
    if ylim is not None:
        ax.set_ylim(ylim)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight=fontweight)

    # z-axis
    if zlabel is not None:
        ax.set_zlabel(zlabel, fontsize=fontsize, fontweight=fontweight)

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
