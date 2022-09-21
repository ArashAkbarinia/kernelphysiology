import numpy as np
import shutil
import os

from matplotlib import pyplot as plt
from skimage import io


def plot_shape_with_pred(preds, which_inds, shapes_conf, colours, in_img_dir, bg=128):
    preds = preds[which_inds]
    fig = plt.figure(figsize=(8, 8))
    j = 1
    rows = int(np.sqrt(len(shapes_conf[which_inds, :])))
    cols = rows
    for param_ind, params in enumerate(shapes_conf[which_inds, :]):
        ax = fig.add_subplot(rows, cols, j)
        j = j + 1
        # img_path = '%s/m_%.4d_n1_%.4d_n2_%.4d_n3_%.4d.png' % (in_img_dir, *params)
        img_path = '%s/a_%.4d_b_%.4d_m_%.4d_n_%.4d_rot_%.4d.png' % (in_img_dir, *params)
        img = io.imread(img_path)
        bg_inds = img == 0
        res_img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        for cind in range(3):
            cchn = res_img[:, :, cind]
            cchn[cchn == 255] = colours[preds[param_ind]][cind]
            cchn[bg_inds] = bg
            res_img[:, :, cind] = cchn
        ax.imshow(res_img)
        ax.axis('off')
    fig.tight_layout()
    return fig


def colour_categorise_shapes(shapes_conf, preds, params, colours, out_dir, out_name, in_img_dir,
                             bg=128):
    unique_params = []
    for i in range(shapes_conf.shape[1]):
        unique_params.append(np.unique(shapes_conf[:, i]))
    l_unique0 = len(unique_params[params[0]])
    l_unique1 = len(unique_params[params[1]])

    tmp_dir = '/tmp/colour_categorise_tmp/'
    os.makedirs(tmp_dir, exist_ok=True)
    cat_fig = plt.figure(figsize=(6 * l_unique1/2, 6 * l_unique0))
    cat_fig_ax_ind = 1
    k = 0
    for i, ri in enumerate(unique_params[params[0]]):
        for j, cj in enumerate(unique_params[params[1]]):
            fig_path = '%simg_i%.2d_j%.2d.png' % (tmp_dir, i, j)
            which_inds = (shapes_conf[:, params[0]] == ri) & (shapes_conf[:, params[1]] == cj)
            fig = plot_shape_with_pred(preds, which_inds, shapes_conf, colours, in_img_dir, bg)
            fig.savefig(fig_path)
            plt.close(fig)
            k = k+1
            if k in [1,3,5]:
                continue
            cat_fig_ax = cat_fig.add_subplot(l_unique0, 3, cat_fig_ax_ind)
            cat_fig_ax_ind = cat_fig_ax_ind + 1
            img = io.imread(fig_path)
            cat_fig_ax.imshow(img)
            cat_fig_ax.axis('off')
    fig_path = '%s/%s.svg' % (out_dir, out_name)
    cat_fig.tight_layout()
    cat_fig.savefig(fig_path)
    shutil.rmtree(tmp_dir)
    return cat_fig


def plot_face_results(ax, similarity, fig_mag=1):
    rows, cols = similarity.shape
    ax.imshow(similarity, vmin=0, vmax=1.0, cmap='PiYG')
    ax.axis('off')

    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            ax.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        ax.spines[side].set_visible(False)

    return ax


def plot_grid_img(cat_inds, which_inds):
    pred_vec = cat_inds[which_inds]
    img_vis = pred_vec.reshape(6, 6) * 1
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img_vis, cmap='PiYG')
    # ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(-.5, img_vis.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, img_vis.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=2)
    fig.tight_layout()
    return fig
