import numpy as np
import glob
import ntpath
import os
from matplotlib import pyplot as plt
import sys

from kernelphysiology.utils import path_utils

csf_dir = "/home/arash/Desktop/projects/csf/"
activations_dir = "%s/data/kernel_activations/" % csf_dir
out_dir = "%s/figures/activations/" % csf_dir
target_size = 256
base_sf = ((target_size / 2) / np.pi)


def process_network(net_name):
    for file_path in sorted(
            glob.glob(os.path.join(activations_dir, net_name) + '/*.pickle')):
        file_name = ntpath.basename(file_path)
        layer_name = 'layer'
        name_parts = file_name.split('_')
        for pind, part in enumerate(name_parts):
            if part == 'layer':
                layer_name += name_parts[pind + 1].replace('.conv2.weight', '')
                break
        png_name = os.path.join(out_dir, net_name,
                                '%s_activation_%s.png' % (layer_name, 'max'))
        if os.path.exists(png_name):
            continue
        print('reading', file_name, layer_name)
        result_mat = path_utils.read_pickle(file_path)
        contrast_activation, xaxis = process_layer(result_mat)
        plot_layer(contrast_activation, xaxis, net_name, layer_name)
    return


def plot_layer(contrast_activation, xaxis, net_name, layer_name):
    report_types = ['lavg', 'lmed', 'lmax', 'ravg', 'rmed', 'rmax', 'avg',
                    'med', 'max']
    for report_key in report_types:
        fig = plot_activations(contrast_activation, report_key, xaxis)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, net_name, '%s_activation_%s.png' % (
            layer_name, report_key)))
        plt.close('all')
    return


def process_layer(result_mat):
    result_mat = np.array(result_mat['results'])
    unique_contrasts = np.unique(result_mat[:, 0])

    contrast_activation = dict()
    for contrast in unique_contrasts:
        val_contrast = extract_contrast(result_mat, contrast)
        unique_frequencies = np.unique(val_contrast[:, 1])
        nkernels = val_contrast[0, 5].shape[0]
        nsamples = val_contrast.shape[0]
        contrast_report = {
            'lavg': np.zeros((nsamples, nkernels)),
            'lmed': np.zeros((nsamples, nkernels)),
            'lmax': np.zeros((nsamples, nkernels)),
            'ravg': np.zeros((nsamples, nkernels)),
            'rmed': np.zeros((nsamples, nkernels)),
            'rmax': np.zeros((nsamples, nkernels)),
        }
        rkeys = list(contrast_report.keys())

        # create a numpy matrix out of all kernels
        for i in range(val_contrast.shape[0]):
            for j in range(len(rkeys)):
                contrast_report[rkeys[j]][i] = val_contrast[i, j + 5]
        # putting left and right together
        contrast_report['avg'] = (contrast_report['lavg'] + contrast_report[
            'ravg']) / 2
        contrast_report['med'] = (contrast_report['lmed'] + contrast_report[
            'rmed']) / 2
        contrast_report['max'] = (contrast_report['lmax'] + contrast_report[
            'rmax']) / 2
        # computing the average of each kernel for all samples
        for rkey, rval in contrast_report.items():
            activation_freq = []
            for freq in unique_frequencies:
                condition = val_contrast[:, 1] == freq
                activation_freq.append(np.mean(rval[condition, :], axis=0))
            contrast_report[rkey] = activation_freq
        contrast_activation[str(contrast)] = contrast_report

    xaxis = [((1 / e) * base_sf) for e in unique_frequencies]
    return contrast_activation, xaxis


def extract_contrast(result_mat, contrast):
    return result_mat[result_mat[:, 0] == contrast, :]


def extract_frequency(result_mat, frequency):
    return result_mat[result_mat[:, 1] == frequency, :]


def plot_activations(activations, report_key, xaxis):
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
            ax.plot(xaxis, toplot, '-x', label=ckey)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if i == 0:
            ax.legend()
            ax.set_xlabel('SF')
            ax.set_ylabel('Activation')
    return fig


if __name__ == "__main__":
    process_network(sys.argv[1])
