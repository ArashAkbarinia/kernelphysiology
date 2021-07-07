import numpy as np
import re
import glob
import ntpath
import os
from matplotlib import pyplot as plt
import sys
from scipy import stats

from kernelphysiology.utils import path_utils
from kernelphysiology.utils.controls import natural_keys

layer_type = 'relu'
csf_dir = "/home/arash/Desktop/projects/csf/"
activations_dir = "%s/data/kernel_activations/%s/" % (csf_dir, layer_type)
fig_out_dir = "%s/figures/activations/" % csf_dir
anl_out_dir = "%s/analysis/activation_corrs/%s/" % (csf_dir, layer_type)
target_size = 256
base_sf = ((target_size / 2) / np.pi)


def get_human_csf(f):
    return 2.6 * (0.0192 + 0.114 * f) * np.exp(-(0.114 * f) ** 1.1)


def get_network_activation(net_name):
    all_layers_maxsf = []
    all_activations = []
    for file_path in sorted(
            glob.glob(os.path.join(activations_dir, net_name) + '/*.pickle'),
            key=natural_keys
    ):
        file_name = ntpath.basename(file_path)
        layer_name = 'layer'
        name_parts = file_name.split('_')
        for pind, part in enumerate(name_parts):
            if part == 'layer':
                layer_name += name_parts[pind + 1].replace('.weight', '')
                layer_name = layer_name.replace('conv', '')
                break
        print('reading', file_name, layer_name)
        result_mat = path_utils.read_pickle(file_path)
        contrast_activation, xvals = process_layer(result_mat)
        all_activations.append(contrast_activation)
        maxsf, header = maxsf_layer(
            contrast_activation, xvals, net_name, layer_name
        )
        all_layers_maxsf.append(maxsf)
    return all_layers_maxsf, all_activations, xvals


def process_network(net_name):
    all_layers_maxsf = []
    for file_path in sorted(
            glob.glob(os.path.join(activations_dir, net_name) + '/*.pickle'),
            key=natural_keys
    ):
        file_name = ntpath.basename(file_path)
        layer_name = 'layer'
        name_parts = file_name.split('_')
        for pind, part in enumerate(name_parts):
            if part == 'layer':
                layer_name += name_parts[pind + 1].replace('.weight', '')
                layer_name = layer_name.replace('conv', '')
                break
        png_name = os.path.join(
            fig_out_dir, net_name, '%s_activation_%s.png' % (layer_name, 'max')
        )
        csv_name = os.path.join(
            anl_out_dir, net_name, '%s_corrs_0.1.csv' % layer_name
        )
        print('reading', file_name, layer_name)
        result_mat = path_utils.read_pickle(file_path)
        contrast_activation, xvals = process_layer(result_mat)
        # if not os.path.exists(png_name):
        #     plot_layer(contrast_activation, xvals, net_name, layer_name)
        # if not os.path.exists(csv_name):
        #     corr_layer(contrast_activation, xvals, net_name, layer_name)
        maxsf, header = maxsf_layer(
            contrast_activation, xvals, net_name, layer_name
        )
        all_layers_maxsf.append(maxsf)
    out_file = os.path.join(
        anl_out_dir, 'peak_activations_avg', '%s_corrs.csv' % net_name
    )
    np.savetxt(
        out_file, np.array(all_layers_maxsf), delimiter=',', header=header
    )
    return


def plot_layer(contrast_activation, xvals, net_name, layer_name):
    report_types = [
        'lavg', 'lmed', 'lmax', 'ravg', 'rmed', 'rmax',
        'avg', 'med', 'max', 'pavg', 'pmed', 'pmax'
    ]
    xaxis = [((1 / e) * base_sf) for e in xvals]
    human_csf = np.array([get_human_csf(f) for f in xaxis])
    for report_key in report_types:
        fig = plot_activations(
            contrast_activation, report_key, xaxis, human_csf
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                fig_out_dir, net_name,
                '%s_activation_%s.png' % (layer_name, report_key)
            )
        )
        plt.close('all')
    return


def corr_layer(contrast_activation, xvals, net_name, layer_name):
    report_types = [
        'lavg', 'lmed', 'lmax', 'ravg', 'rmed', 'rmax',
        'avg', 'med', 'max', 'pavg', 'pmed', 'pmax'
    ]
    xaxis = [((1 / e) * base_sf) for e in xvals]
    human_csf = np.array([get_human_csf(f) for f in xaxis])

    contrast_keys = list(contrast_activation.keys())
    reports_cross = dict()
    for ckey in contrast_keys:
        reports_cross[ckey] = []

    for report_key in report_types:
        corrs = corr_activations(contrast_activation, report_key, human_csf)
        for ckey, cval in corrs.items():
            reports_cross[ckey].append(cval)
    header = ','.join(e for e in report_types)
    for ckey, cval in reports_cross.items():
        out_file = os.path.join(
            anl_out_dir, net_name, '%s_corrs_%s.csv' % (layer_name, ckey)
        )
        np.savetxt(out_file, np.array(cval).T, delimiter=',', header=header)
    return


def interpolate_all_sfs(xvals, yvals, target_size=256):
    base_sf = ((target_size / 2) / np.pi)
    new_xs = [base_sf / e for e in np.arange(1, 129, 0.5)]
    new_ys = np.interp(new_xs, xvals, yvals)
    return new_xs, new_ys


def maxsf_layer(contrast_activation, xvals, net_name, layer_name):
    report_types = ['pavg', 'pmed', 'pmax']
    base_sf = ((target_size / 2) / np.pi)
    newxs = [base_sf / e for e in np.arange(1, 129, 0.5)]

    xaxis = [((1 / e) * base_sf) for e in newxs]
    human_csf = np.array([get_human_csf(f) for f in xaxis])

    headers = []
    max_sfs = []
    newxs = np.array(newxs, dtype='float64')
    xvals = np.array(xvals, dtype='float64')
    for i, report_key in enumerate(report_types):
        maxsf = max_activations(contrast_activation, report_key)

        all_newys = np.zeros(
            (len(maxsf.keys()), *human_csf.shape)
        )
        for j, (ckey, cval) in enumerate(maxsf.items()):
            yvals = cval / cval.max()
            newys = np.interp(newxs, xvals, yvals)
            all_newys[j] = newys
            # p_corr, r_corr = stats.pearsonr(human_csf, newys)
            # max_sfs.append(p_corr)
            # headers.append(report_key + ckey)
        p_corr, r_corr = stats.pearsonr(human_csf, all_newys.mean(axis=0))
        max_sfs.append(p_corr)
        headers.append(report_key + 'allcs')
    header = ','.join(e for e in headers)

    # plotting the peack frequencies
    # fig = plt.figure(figsize=(18, 6))
    # for i, report_key in enumerate(report_types):
    #     maxsf = max_activations(contrast_activation, report_key)
    #
    #     ax = fig.add_subplot(1, 3, i + 1)
    #     for ckey, cval in maxsf.items():
    #         toplot = cval / cval.max()
    #         p_corr, r_corr = stats.pearsonr(human_csf, np.array(toplot))
    #         ax.plot(xaxis, toplot, '-x', label=ckey + ' - %.2f' % p_corr)
    #     ax.plot(xaxis, human_csf, '--', color='black')
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])
    #
    #     ax.legend()
    #     ax.set_xlabel('SF')
    #     ax.set_ylabel('#Kernels')
    #     ax.set_title(report_key)
    #
    # fig.tight_layout()
    # fig.savefig(
    #     os.path.join(
    #         fig_out_dir, 'peak_activations', net_name,
    #         '%s_peakactivation.png' % (layer_name)
    #     )
    # )
    # plt.close('all')
    return max_sfs, header


def process_layer(result_mat):
    result_mat = np.array(result_mat['results'])
    unique_contrasts = np.unique(result_mat[:, 0])

    contrast_activation = dict()
    for contrast in unique_contrasts:
        val_contrast = extract_contrast(result_mat, contrast)
        unique_frequencies = np.unique(val_contrast[:, 1])
        nkernels = val_contrast[0, 5].shape[0]
        nsamples = val_contrast.shape[0]
        report = {
            'lavg': np.zeros((nsamples, nkernels)),
            'lmed': np.zeros((nsamples, nkernels)),
            'lmax': np.zeros((nsamples, nkernels)),
            'ravg': np.zeros((nsamples, nkernels)),
            'rmed': np.zeros((nsamples, nkernels)),
            'rmax': np.zeros((nsamples, nkernels)),
        }
        rkeys = list(report.keys())

        # create a numpy matrix out of all kernels
        for i in range(val_contrast.shape[0]):
            for j in range(len(rkeys)):
                report[rkeys[j]][i] = val_contrast[i, j + 5]
        # putting left and right together
        for pk in ['avg', 'med', 'max']:
            report[pk] = (report['l' + pk] + report['r' + pk]) / 2

        # activity at the side of where stimuli was presented
        # sidef == 0 means the right side
        cols = val_contrast[:, 4] == 0
        for pk in ['avg', 'med', 'max']:
            report['p' + pk] = report['l' + pk]
            report['p' + pk][cols, :] = report['r' + pk][cols, :]

        # computing the average of each kernel for all samples
        for rkey, rval in report.items():
            activation_freq = []
            for freq in unique_frequencies:
                cols = val_contrast[:, 1] == freq
                activation_freq.append(np.mean(rval[cols, :], axis=0))
            report[rkey] = activation_freq
        contrast_activation[str(contrast)] = report

    xvals = unique_frequencies
    return contrast_activation, xvals


def extract_contrast(result_mat, contrast):
    return result_mat[result_mat[:, 0] == contrast, :]


def extract_frequency(result_mat, frequency):
    return result_mat[result_mat[:, 1] == frequency, :]


def plot_activations(activations, report_key, xaxis, human_csf):
    contrast_keys = list(activations.keys())
    nkernels = activations[contrast_keys[0]][report_key][0].shape[0]
    rows = int(nkernels / 8)
    cols = 8
    fig = plt.figure(figsize=(18, rows * 2))
    for i in range(nkernels):
        ax = fig.add_subplot(rows, cols, i + 1)
        for ckey in contrast_keys:
            toplot = []
            for freq in activations[ckey][report_key]:
                toplot.append(freq[i])
            p_corr, r_corr = stats.pearsonr(human_csf, np.array(toplot))
            ax.plot(xaxis, toplot, '-x', label=ckey + ' - %.2f' % p_corr)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.legend()
        ax.set_xlabel('SF')
        ax.set_ylabel('Activation')
    return fig


def corr_activations(activations, report_key, human_csf):
    contrast_keys = list(activations.keys())
    nkernels = activations[contrast_keys[0]][report_key][0].shape[0]

    corrs = dict()
    for ckey in contrast_keys:
        corrs[ckey] = []
    for i in range(nkernels):
        for ckey in contrast_keys:
            net_csf = []
            for freq in activations[ckey][report_key]:
                net_csf.append(freq[i])
            p_corr, r_corr = stats.pearsonr(human_csf, np.array(net_csf))
            corrs[ckey].append(p_corr)

    return corrs


def max_activations(activations, report_key):
    contrast_keys = list(activations.keys())
    nkernels = activations[contrast_keys[0]][report_key][0].shape[0]
    nfreqs = len(activations[contrast_keys[0]][report_key])

    maxsfs = dict()
    for ckey in contrast_keys:
        maxsfs[ckey] = np.zeros((nfreqs))
    for i in range(nkernels):
        for ckey in contrast_keys:
            net_csf = []
            for freq in activations[ckey][report_key]:
                net_csf.append(freq[i])
            max_ind = np.array(net_csf).argmax()
            maxsfs[ckey][max_ind] += 1

    return maxsfs


if __name__ == "__main__":
    process_network(sys.argv[1])
